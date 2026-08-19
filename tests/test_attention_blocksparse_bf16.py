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


def build_paged_kvcache(
    num_batch, num_seq_kv, num_head_kv, head_dim, block_size, kv_layout, device="cuda"
):
    """Allocate a paged BF16 KV cache with a shuffled page table.

    Returns (kcache, vcache, block_ids, seqlens_kvcache).  ``hnd`` produces a
    stride-transformed view with the same logical shape, which is what the
    kernel's layout support is about.
    """
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


def build_ordered_list_from_mask(block_mask, shuffle=False, seed=0):
    """Turn a bool block mask into CoSA's ordered int32 list, padded with -1.

    ``shuffle`` permutes each row so that the kernel is exercised on an order
    that is not the logical one; the selected set is unchanged.
    """
    num_batch, num_head_q, num_tile_m, num_tile_kv = block_mask.shape
    ordered = torch.full(
        (num_batch, num_head_q, num_tile_m, num_tile_kv),
        -1,
        dtype=torch.int32,
        device=block_mask.device,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for i in range(num_batch):
        for h in range(num_head_q):
            for t in range(num_tile_m):
                idx = block_mask[i, h, t].nonzero().flatten().cpu()
                if shuffle and idx.numel() > 1:
                    idx = idx[torch.randperm(idx.numel(), generator=generator)]
                ordered[i, h, t, : idx.numel()] = idx.to(torch.int32).to(ordered.device)
    return ordered


def naive_attn_with_kvcache_cosa_bf16(
    q,
    kcache,
    vcache,
    seqlens_kvcache,
    block_ids,
    ordered_block_indices,
    threshold,
):
    """FP32 reference for CoSA, replaying the kernel's tile order and skip rule.

    The skip decision depends on the running row max, which depends on how many
    tiles have been processed so far, so the reference has to walk each Q tile's
    list in the given order rather than reduce over a set.  Once the processed
    set is known, the attention itself is an ordinary exact softmax over it.

    Returns (output, num_tile_skipped) where the count is over threshold skips
    only, summed over every (batch, head, q-tile).
    """
    log2e = 1.4426950408889634
    num_batch, num_seq_q, num_head_q, head_dim = q.shape
    num_head_kv = kcache.size(2)
    num_group = num_head_q // num_head_kv
    num_tile_kv_in_list = ordered_block_indices.size(3)
    output = torch.empty_like(q)
    num_tile_skipped = 0

    for i in range(num_batch):
        num_seq_kv = int(seqlens_kvcache[i].item())
        start_seq_q = num_seq_kv - num_seq_q
        bq = q[i].transpose(0, 1).float()
        bk, bv = gather_kv(kcache, vcache, block_ids, num_seq_kv, i, num_head_kv, head_dim)
        bk = bk.repeat_interleave(num_group, dim=0)
        bv = bv.repeat_interleave(num_group, dim=0)

        # threshold == 0 gives -inf, i.e. the comparison is never true and nothing is skipped.
        log2_threshold = (
            math.log2(min(threshold / num_seq_kv, 0.1)) if threshold > 0 else -float("inf")
        )

        num_tile_m = (num_seq_q + BSA_BLOCK - 1) // BSA_BLOCK
        for h in range(num_head_q):
            for t in range(num_tile_m):
                row_lo = t * BSA_BLOCK
                row_hi = min(row_lo + BSA_BLOCK, num_seq_q)
                num_tile_full = (start_seq_q + row_lo) // BSA_BLOCK

                running_max = torch.full(
                    (row_hi - row_lo,), -float("inf"), device=q.device, dtype=torch.float32
                )
                scores_selected = []
                v_selected = []

                for j in range(num_tile_kv_in_list):
                    itile_kv = int(ordered_block_indices[i, h, t, j].item())
                    if itile_kv < 0:
                        break
                    col_lo = itile_kv * BSA_BLOCK
                    col_hi = min(col_lo + BSA_BLOCK, num_seq_kv)

                    scores = torch.matmul(
                        bq[h, row_lo:row_hi], bk[h, col_lo:col_hi].transpose(-2, -1)
                    ) / math.sqrt(head_dim)

                    if itile_kv >= num_tile_full:
                        # Diagonal tile: element-wise causal mask, exempt from the skip vote.
                        irow = start_seq_q + torch.arange(
                            row_lo, row_hi, device=q.device
                        ).unsqueeze(1)
                        icol = torch.arange(col_lo, col_hi, device=q.device).unsqueeze(0)
                        scores = scores.masked_fill(icol > irow, float("-inf"))
                    else:
                        row_max = scores.max(dim=-1).values * log2e
                        if bool(((row_max - running_max) < log2_threshold).all().item()):
                            num_tile_skipped += 1
                            continue

                    running_max = torch.maximum(running_max, scores.max(dim=-1).values * log2e)
                    scores_selected.append(scores)
                    v_selected.append(bv[h, col_lo:col_hi])

                if not scores_selected:
                    output[i, row_lo:row_hi, h] = float("nan")
                    continue

                weights = torch.softmax(torch.cat(scores_selected, dim=-1), dim=-1)
                acc = torch.matmul(weights, torch.cat(v_selected, dim=0))
                output[i, row_lo:row_hi, h] = acc.to(q.dtype)

    return output, num_tile_skipped


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


# ====================================================================
# CoSA mode
# ====================================================================


@pytest.mark.parametrize("num_batch", [1, 2])
@pytest.mark.parametrize("num_seq", [1024])
@pytest.mark.parametrize("num_head_q,num_head_kv", [(4, 1), (2, 2)])
@pytest.mark.parametrize("block_size", [32, 64])
@pytest.mark.parametrize("list_mode", ["full", "sparse", "shuffled"])
@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_blocksparse_bf16_cosa(
    num_batch,
    num_seq,
    num_head_q,
    num_head_kv,
    block_size,
    list_mode,
    threshold,
):
    """Ordered-list access over unit-variance data.

    With inputs like these the tiles sit within a couple of base-2 units of each
    other while the threshold is 10 units down, so threshold=1.0 runs the vote and
    its two named barriers on every non-diagonal tile without ever skipping one.
    That is the point here — the skip itself needs a deliberately spread-out
    construction and is covered by
    ``test_blocksparse_bf16_cosa_threshold_skips_tiles``.

    Sequence lengths are tile-aligned on purpose: the vote covers all 128 rows of
    a Q tile, including rows past the end of a non-aligned sequence, so only
    aligned lengths let the reference reproduce a firing decision exactly.
    """
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)

    head_dim = 128
    device = "cuda"

    q = torch.randn(
        num_batch, num_seq, num_head_q, head_dim, dtype=torch.bfloat16, device=device
    ) / math.sqrt(head_dim)
    cu_seqlens_q = make_cu_seqlens(num_batch, num_seq, device)
    kcache, vcache, block_ids, seqlens_kvcache = build_paged_kvcache(
        num_batch, num_seq, num_head_kv, head_dim, block_size, "nhd"
    )

    ntiles = (num_seq + BSA_BLOCK - 1) // BSA_BLOCK
    skip_ratio = 0.0 if list_mode == "full" else 0.5
    block_mask = generate_block_sparse_mask(
        num_batch, num_head_q, ntiles, ntiles, skip_ratio, causal=True, device=device
    )
    ordered = build_ordered_list_from_mask(block_mask, shuffle=(list_mode == "shuffled"))

    gt, _ = naive_attn_with_kvcache_cosa_bf16(
        q, kcache, vcache, seqlens_kvcache, block_ids, ordered, threshold
    )
    gt = gt.reshape(-1, num_head_q, head_dim)

    my = hpc.attention_with_kvcache_blocksparse_prefill_bf16(
        q.reshape(-1, num_head_q, head_dim),
        kcache,
        vcache,
        cu_seqlens_q,
        block_ids,
        seqlens_kvcache,
        num_seq,
        enable_cosa=True,
        ordered_block_indices=ordered,
        threshold=threshold,
    )

    assert allclose(gt, my, atol=0.016)


@pytest.mark.parametrize("num_seq", [1500])
def test_blocksparse_bf16_cosa_unaligned_no_skip(num_seq):
    """Non-tile-aligned lengths with threshold=0: ordered access, no skip decision."""
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    num_batch, num_head_q, num_head_kv, head_dim, block_size = 2, 4, 1, 128, 64
    device = "cuda"

    q = torch.randn(
        num_batch, num_seq, num_head_q, head_dim, dtype=torch.bfloat16, device=device
    ) / math.sqrt(head_dim)
    cu_seqlens_q = make_cu_seqlens(num_batch, num_seq, device)
    kcache, vcache, block_ids, seqlens_kvcache = build_paged_kvcache(
        num_batch, num_seq, num_head_kv, head_dim, block_size, "nhd"
    )

    ntiles = (num_seq + BSA_BLOCK - 1) // BSA_BLOCK
    block_mask = generate_block_sparse_mask(
        num_batch, num_head_q, ntiles, ntiles, 0.5, causal=True, device=device
    )
    ordered = build_ordered_list_from_mask(block_mask, shuffle=True)

    gt, num_skipped = naive_attn_with_kvcache_cosa_bf16(
        q, kcache, vcache, seqlens_kvcache, block_ids, ordered, 0.0
    )
    assert num_skipped == 0

    my = hpc.attention_with_kvcache_blocksparse_prefill_bf16(
        q.reshape(-1, num_head_q, head_dim),
        kcache,
        vcache,
        cu_seqlens_q,
        block_ids,
        seqlens_kvcache,
        num_seq,
        enable_cosa=True,
        ordered_block_indices=ordered,
        threshold=0.0,
    )

    assert allclose(gt.reshape(-1, num_head_q, head_dim), my, atol=0.016)


def test_blocksparse_bf16_cosa_matches_mask_path_without_skip():
    """threshold=0 makes CoSA a pure reordering, so it must match the mask path.

    This compares two kernel modes against each other, so it fails if either the
    ordered-list read or the mask compaction selects the wrong tiles, without
    relying on the Python reference at all.
    """
    torch.manual_seed(99)
    torch.cuda.manual_seed(99)

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
    block_mask = generate_block_sparse_mask(
        num_batch, num_head_q, ntiles, ntiles, 0.5, causal=True, device=device
    )
    ordered = build_ordered_list_from_mask(block_mask, shuffle=False)

    q_flat = q.reshape(-1, num_head_q, head_dim)
    args = (q_flat, kcache, vcache, cu_seqlens_q, block_ids, seqlens_kvcache, num_seq)
    via_mask = hpc.attention_with_kvcache_blocksparse_prefill_bf16(
        *args, block_mask=block_mask.to(torch.uint8).contiguous()
    )
    via_cosa = hpc.attention_with_kvcache_blocksparse_prefill_bf16(
        *args, enable_cosa=True, ordered_block_indices=ordered, threshold=0.0
    )

    assert allclose(via_mask, via_cosa, atol=0.016)


def build_cosa_skip_case(device="cuda", seed=2024):
    """Build inputs where the threshold skip both fires robustly and is observable.

    Those two goals pull against each other: the vote passes only when every row
    of the Q tile is at least ``log2(threshold / num_seq_kv)`` below the running
    max, and a tile that far down contributes almost nothing, so skipping it would
    barely move the output.  The way out is that the vote reads scores only, never
    V.  So K sets the decision — the pages behind KV tile 0 are amplified and the
    rest attenuated, putting the cold tiles a comfortable margin below the
    threshold rather than right at it — and V makes the consequence visible, by
    giving the cold pages a large constant value.  Skipping or not skipping them
    then differs by far more than the comparison tolerance.

    Returns (q, kcache, vcache, block_ids, seqlens_kvcache, cu_seqlens_q,
    ordered_block_indices, num_seq, num_head_q, head_dim).
    """
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    num_batch, num_seq, num_head_q, num_head_kv, head_dim, block_size = 1, 1024, 2, 1, 128, 64

    q = torch.randn(
        num_batch, num_seq, num_head_q, head_dim, dtype=torch.bfloat16, device=device
    ) / math.sqrt(head_dim)
    cu_seqlens_q = make_cu_seqlens(num_batch, num_seq, device)
    kcache, vcache, block_ids, seqlens_kvcache = build_paged_kvcache(
        num_batch, num_seq, num_head_kv, head_dim, block_size, "nhd", device=device
    )

    pages_per_tile = BSA_BLOCK // block_size
    for i in range(num_batch):
        hot_pages = block_ids[i, :pages_per_tile].long()
        cold_pages = block_ids[i, pages_per_tile:].long()
        kcache[hot_pages] *= 64.0
        kcache[cold_pages] *= 0.02
        vcache[cold_pages] = 16.0

    ntiles = num_seq // BSA_BLOCK
    ordered = torch.full((num_batch, num_head_q, ntiles, ntiles), -1, dtype=torch.int32)
    for t in range(ntiles):
        ordered[:, :, t, : t + 1] = torch.arange(t + 1, dtype=torch.int32)
    ordered = ordered.to(device)

    return (
        q,
        kcache,
        vcache,
        block_ids,
        seqlens_kvcache,
        cu_seqlens_q,
        ordered,
        num_seq,
        num_head_q,
        head_dim,
    )


def test_blocksparse_bf16_cosa_threshold_skips_tiles():
    """The kernel must skip the tiles the reference skips, observably.

    Matching the skipping reference is only meaningful because the two references
    are far apart: the same data with threshold=0 gives a visibly different
    result, and the kernel is checked against that one too.  So this fails both if
    the kernel skips nothing and if it skips the wrong tiles.
    """
    (
        q,
        kcache,
        vcache,
        block_ids,
        seqlens_kvcache,
        cu_seqlens_q,
        ordered,
        num_seq,
        num_head_q,
        head_dim,
    ) = build_cosa_skip_case()
    threshold = 1.0

    ref_skip, num_skipped = naive_attn_with_kvcache_cosa_bf16(
        q, kcache, vcache, seqlens_kvcache, block_ids, ordered, threshold
    )
    ref_noskip, num_skipped_zero = naive_attn_with_kvcache_cosa_bf16(
        q, kcache, vcache, seqlens_kvcache, block_ids, ordered, 0.0
    )
    assert num_skipped > 0, "construction no longer triggers the threshold skip"
    assert num_skipped_zero == 0
    separation = (ref_skip.float() - ref_noskip.float()).abs().max().item()
    assert separation > 0.1, f"skipping is not observable in the output (max diff {separation})"

    q_flat = q.reshape(-1, num_head_q, head_dim)
    args = (q_flat, kcache, vcache, cu_seqlens_q, block_ids, seqlens_kvcache, num_seq)
    my_skip = hpc.attention_with_kvcache_blocksparse_prefill_bf16(
        *args, enable_cosa=True, ordered_block_indices=ordered, threshold=threshold
    )
    my_noskip = hpc.attention_with_kvcache_blocksparse_prefill_bf16(
        *args, enable_cosa=True, ordered_block_indices=ordered, threshold=0.0
    )

    assert allclose(ref_skip.reshape(-1, num_head_q, head_dim), my_skip, atol=0.03)
    assert allclose(ref_noskip.reshape(-1, num_head_q, head_dim), my_noskip, atol=0.03)


# ====================================================================
# Argument validation
# ====================================================================


def _cosa_validation_inputs():
    num_batch, num_seq, num_head_q, num_head_kv, head_dim, block_size = 1, 256, 2, 1, 128, 64
    device = "cuda"
    q = torch.randn(
        num_batch * num_seq, num_head_q, head_dim, dtype=torch.bfloat16, device=device
    ) / math.sqrt(head_dim)
    cu_seqlens_q = make_cu_seqlens(num_batch, num_seq, device)
    kcache, vcache, block_ids, seqlens_kvcache = build_paged_kvcache(
        num_batch, num_seq, num_head_kv, head_dim, block_size, "nhd"
    )
    ntiles = num_seq // BSA_BLOCK
    ordered = torch.zeros(num_batch, num_head_q, ntiles, ntiles, dtype=torch.int32, device=device)
    block_mask = torch.ones(num_batch, num_head_q, ntiles, ntiles, dtype=torch.uint8, device=device)
    args = (q, kcache, vcache, cu_seqlens_q, block_ids, seqlens_kvcache, num_seq)
    return args, ordered, block_mask


def test_cosa_requires_ordered_list_and_threshold():
    args, ordered, _ = _cosa_validation_inputs()

    with pytest.raises(RuntimeError, match="requires ordered_block_indices"):
        hpc.attention_with_kvcache_blocksparse_prefill_bf16(*args, enable_cosa=True, threshold=1.0)

    with pytest.raises(RuntimeError, match="requires threshold"):
        hpc.attention_with_kvcache_blocksparse_prefill_bf16(
            *args, enable_cosa=True, ordered_block_indices=ordered
        )


def test_cosa_rejects_block_mask():
    args, ordered, block_mask = _cosa_validation_inputs()

    with pytest.raises(RuntimeError, match="incompatible with block_mask"):
        hpc.attention_with_kvcache_blocksparse_prefill_bf16(
            *args,
            block_mask=block_mask,
            enable_cosa=True,
            ordered_block_indices=ordered,
            threshold=1.0,
        )


def test_cosa_args_rejected_when_disabled():
    args, ordered, _ = _cosa_validation_inputs()

    with pytest.raises(RuntimeError, match="only accepted with enable_cosa"):
        hpc.attention_with_kvcache_blocksparse_prefill_bf16(*args, ordered_block_indices=ordered)

    with pytest.raises(RuntimeError, match="only accepted with enable_cosa"):
        hpc.attention_with_kvcache_blocksparse_prefill_bf16(*args, threshold=1.0)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda t: t.float(), "dtype must be int32"),
        (lambda t: t[..., ::2], "must be contiguous"),
        (lambda t: t[:, :1], "must have shape"),
        (lambda t: t[..., :0].contiguous(), "Kb dim must be > 0"),
    ],
)
def test_cosa_rejects_bad_ordered_list(mutate, match):
    args, ordered, _ = _cosa_validation_inputs()

    with pytest.raises(RuntimeError, match=match):
        hpc.attention_with_kvcache_blocksparse_prefill_bf16(
            *args, enable_cosa=True, ordered_block_indices=mutate(ordered), threshold=1.0
        )


@pytest.mark.parametrize("bad_threshold", [-1.0, float("inf"), float("nan")])
def test_cosa_rejects_bad_threshold(bad_threshold):
    args, ordered, _ = _cosa_validation_inputs()

    with pytest.raises(RuntimeError, match="threshold must be finite"):
        hpc.attention_with_kvcache_blocksparse_prefill_bf16(
            *args, enable_cosa=True, ordered_block_indices=ordered, threshold=bad_threshold
        )
