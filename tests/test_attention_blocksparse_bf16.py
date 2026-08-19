import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import torch

import hpc
from utils import allclose

BSA_BLOCK = 128


# ====================================================================
# Shared helpers
# ====================================================================


def generate_block_sparse_mask(batch, heads, nrow, ncol, skip_ratio, causal=True, device="cuda"):
    """Block-level sparse mask.  True = attend.

    The causal diagonal block is forced on so that no Q-tile ends up with zero
    active KV tiles (which would make softmax(all -inf) = NaN).
    """
    mask = torch.rand(batch, heads, nrow, ncol, device=device) >= skip_ratio

    row_idx = torch.arange(nrow, device=device).view(nrow, 1)
    col_idx = torch.arange(ncol, device=device).view(1, ncol)

    if causal:
        causal_boundary = row_idx + (ncol - nrow)
        valid_causal = col_idx <= causal_boundary
        mask = mask & valid_causal
        diag_col = torch.clamp(causal_boundary, max=ncol - 1)
        mask = mask | (col_idx == diag_col)

    return mask


def build_paged_kvcache(num_batch, num_seq_kv, num_head_kv, head_dim, block_size, kv_layout):
    """Allocate a paged BF16 KV cache with a shuffled page table.

    Returns (kcache, vcache, block_ids, seqlens_kvcache).  ``hnd`` produces a
    stride-transformed view with the same logical shape, which is what the
    kernel's layout support is about.
    """
    device = "cuda"
    seqlens_kvcache = torch.full((num_batch,), num_seq_kv, dtype=torch.int32, device=device)
    kvcache_blocks = (seqlens_kvcache + block_size - 1) // block_size
    total_kb = int(kvcache_blocks.sum().item())
    max_kb = int(kvcache_blocks.max().item())
    max_num_blocks = total_kb * 2

    kvcache = torch.randn(
        max_num_blocks,
        2,
        block_size,
        num_head_kv,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    if kv_layout == "hnd":
        kcache = kvcache[:, 0].transpose(1, 2).contiguous().transpose(1, 2)
        vcache = kvcache[:, 1].transpose(1, 2).contiguous().transpose(1, 2)
    else:
        kcache = kvcache[:, 0]
        vcache = kvcache[:, 1]

    packed_ids = torch.randperm(max_num_blocks, device=device)[:total_kb].to(torch.int32)
    block_ids = torch.zeros(num_batch, max_kb, dtype=torch.int32, device=device)
    cu = 0
    for i in range(num_batch):
        nb = int(kvcache_blocks[i].item())
        block_ids[i, :nb] = packed_ids[cu : cu + nb]
        cu += nb

    return kcache, vcache, block_ids, seqlens_kvcache


def gather_kv(kcache, vcache, block_ids, num_seq_kv, batch_idx, num_head_kv, head_dim):
    """Materialise the logical [num_head_kv, num_seq_kv, head_dim] K/V for one request."""
    block_size = kcache.size(1)
    nb = (num_seq_kv + block_size - 1) // block_size
    blk = block_ids[batch_idx, :nb]
    k = kcache[blk].reshape(-1, num_head_kv, head_dim).transpose(0, 1)[:, :num_seq_kv, :].float()
    v = vcache[blk].reshape(-1, num_head_kv, head_dim).transpose(0, 1)[:, :num_seq_kv, :].float()
    return k, v


def naive_attn_with_kvcache_blocksparse_bf16(
    q,
    kcache,
    vcache,
    seqlens_kvcache,
    block_ids,
    block_mask=None,
):
    """FP32 reference: causal block-sparse attention over a paged KV cache.

    ``q`` is [num_batch, num_seq_q, num_head_q, head_dim].  ``block_mask`` is a
    bool tensor [num_batch, num_head_q, num_tile_m, num_tile_kv] at BSA_BLOCK
    granularity; None means dense (causal only).
    """
    num_batch, num_seq_q, num_head_q, head_dim = q.shape
    num_head_kv = kcache.size(2)
    num_group = num_head_q // num_head_kv
    output = torch.empty_like(q)

    for i in range(num_batch):
        num_seq_kv = int(seqlens_kvcache[i].item())
        bq = q[i].transpose(0, 1).float()
        bk, bv = gather_kv(kcache, vcache, block_ids, num_seq_kv, i, num_head_kv, head_dim)
        bk = bk.repeat_interleave(num_group, dim=0)
        bv = bv.repeat_interleave(num_group, dim=0)

        scores = torch.matmul(bq, bk.transpose(-2, -1)) / math.sqrt(head_dim)

        if block_mask is not None:
            elem_mask = block_mask[i].repeat_interleave(BSA_BLOCK, dim=-2)[:, :num_seq_q, :]
            elem_mask = elem_mask.repeat_interleave(BSA_BLOCK, dim=-1)[:, :, :num_seq_kv]
            scores = scores.masked_fill(~elem_mask, float("-inf"))

        causal = torch.tril(torch.ones(num_seq_kv, num_seq_kv, device=q.device, dtype=torch.bool))[
            (num_seq_kv - num_seq_q) :, :
        ].unsqueeze(0)
        scores = scores.masked_fill(~causal, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        output[i] = torch.matmul(weights, bv).transpose(0, 1).to(q.dtype)

    return output


def make_cu_seqlens(num_batch, num_seq_q, device="cuda"):
    seqlens_q = torch.full((num_batch,), num_seq_q, dtype=torch.int32, device=device)
    cu = torch.zeros(num_batch + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(seqlens_q, dim=0)
    return cu.to(torch.int32)


# ====================================================================
# Plain BF16 block-sparse prefill
# ====================================================================


@pytest.mark.parametrize("num_batch", [1, 3])
@pytest.mark.parametrize("num_seq", [1024, 1500])
@pytest.mark.parametrize("num_head_q,num_head_kv", [(4, 1)])
@pytest.mark.parametrize("block_size", [32, 64])
@pytest.mark.parametrize("mask_mode", ["none", "dense", "sparse"])
@pytest.mark.parametrize("kv_layout", ["nhd", "hnd"])
@pytest.mark.parametrize("use_output", [False, True])
def test_blocksparse_bf16(
    num_batch,
    num_seq,
    num_head_q,
    num_head_kv,
    block_size,
    mask_mode,
    kv_layout,
    use_output,
):
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)

    head_dim = 128
    device = "cuda"

    q = torch.randn(
        num_batch, num_seq, num_head_q, head_dim, dtype=torch.bfloat16, device=device
    ) / math.sqrt(head_dim)
    cu_seqlens_q = make_cu_seqlens(num_batch, num_seq, device)
    kcache, vcache, block_ids, seqlens_kvcache = build_paged_kvcache(
        num_batch, num_seq, num_head_kv, head_dim, block_size, kv_layout
    )

    ntiles = (num_seq + BSA_BLOCK - 1) // BSA_BLOCK
    if mask_mode == "none":
        block_mask = None
        block_mask_u8 = None
    else:
        skip_ratio = 0.0 if mask_mode == "dense" else 0.5
        block_mask = generate_block_sparse_mask(
            num_batch, num_head_q, ntiles, ntiles, skip_ratio, causal=True, device=device
        )
        block_mask_u8 = block_mask.to(torch.uint8).contiguous()

    gt = naive_attn_with_kvcache_blocksparse_bf16(
        q, kcache, vcache, seqlens_kvcache, block_ids, block_mask=block_mask
    ).reshape(-1, num_head_q, head_dim)

    q_flat = q.reshape(-1, num_head_q, head_dim)
    out = torch.empty_like(q_flat) if use_output else None
    my = hpc.attention_with_kvcache_blocksparse_prefill_bf16(
        q_flat,
        kcache,
        vcache,
        cu_seqlens_q,
        block_ids,
        seqlens_kvcache,
        num_seq,
        block_mask=block_mask_u8,
        output=out,
    )
    if use_output:
        assert my.data_ptr() == out.data_ptr()

    assert allclose(gt, my, atol=0.016)


def test_blocksparse_bf16_deterministic():
    """Same input twice must give bit-identical output."""
    torch.manual_seed(41)
    torch.cuda.manual_seed(41)

    num_batch, num_seq, num_head_q, num_head_kv, head_dim, block_size = 2, 1024, 4, 1, 128, 64
    device = "cuda"

    q = torch.randn(
        num_batch, num_seq, num_head_q, head_dim, dtype=torch.bfloat16, device=device
    ) / math.sqrt(head_dim)
    cu_seqlens_q = make_cu_seqlens(num_batch, num_seq, device)
    kcache, vcache, block_ids, seqlens_kvcache = build_paged_kvcache(
        num_batch, num_seq, num_head_kv, head_dim, block_size, "nhd"
    )
    ntiles = (num_seq + BSA_BLOCK - 1) // BSA_BLOCK
    block_mask_u8 = (
        generate_block_sparse_mask(
            num_batch, num_head_q, ntiles, ntiles, 0.5, causal=True, device=device
        )
        .to(torch.uint8)
        .contiguous()
    )

    q_flat = q.reshape(-1, num_head_q, head_dim)
    args = (q_flat, kcache, vcache, cu_seqlens_q, block_ids, seqlens_kvcache, num_seq)
    first = hpc.attention_with_kvcache_blocksparse_prefill_bf16(*args, block_mask=block_mask_u8)
    second = hpc.attention_with_kvcache_blocksparse_prefill_bf16(*args, block_mask=block_mask_u8)

    assert torch.equal(first, second)
