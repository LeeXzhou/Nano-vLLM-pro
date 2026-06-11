import torch
import triton
import triton.language as tl


@triton.jit
def _rotary_embedding_kernel(
    Q_ptr,
    Q_stride_0,
    Q_stride_1,
    Q_stride_2,
    K_ptr,
    K_stride_0,
    K_stride_1,
    K_stride_2,
    COS_PTR,
    SIN_PTR,
    cos_stride,
    sin_stride,
    positions_ptr,
    ROTARY_DIM_HALF: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    pos = tl.load(positions_ptr + token_idx)

    # Load cos/sin for this position
    half_offsets = tl.arange(0, BLOCK_SIZE)
    mask = half_offsets < ROTARY_DIM_HALF

    cos_ptrs = COS_PTR + pos * cos_stride + half_offsets
    sin_ptrs = SIN_PTR + pos * sin_stride + half_offsets
    cos = tl.load(cos_ptrs, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Load Q first half and second half
    # Q_stride_1 is the head stride (head_dim), Q_stride_2 is element stride (1)
    q_base = Q_ptr + token_idx * Q_stride_0 + head_idx * Q_stride_1
    q1_ptrs = q_base + half_offsets
    q2_ptrs = q_base + ROTARY_DIM_HALF + half_offsets
    q1 = tl.load(q1_ptrs, mask=mask, other=0.0).to(tl.float32)
    q2 = tl.load(q2_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Apply rotation
    new_q1 = q1 * cos - q2 * sin
    new_q2 = q2 * cos + q1 * sin

    # Store Q in-place
    tl.store(q1_ptrs, new_q1, mask=mask)
    tl.store(q2_ptrs, new_q2, mask=mask)

    # Load K first half and second half
    # K_stride_1 is the head stride (head_dim), K_stride_2 is element stride (1)
    k_base = K_ptr + token_idx * K_stride_0 + head_idx * K_stride_1
    k1_ptrs = k_base + half_offsets
    k2_ptrs = k_base + ROTARY_DIM_HALF + half_offsets
    k1 = tl.load(k1_ptrs, mask=mask, other=0.0).to(tl.float32)
    k2 = tl.load(k2_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Apply rotation
    new_k1 = k1 * cos - k2 * sin
    new_k2 = k2 * cos + k1 * sin

    # Store K in-place
    tl.store(k1_ptrs, new_k1, mask=mask)
    tl.store(k2_ptrs, new_k2, mask=mask)


def triton_apply_rotary_emb(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Rotary Embedding using Triton.

    Applies RoPE in-place on Q and K tensors.
    query/key shape: [num_tokens, num_heads, head_dim]
    cos_sin_cache shape: [max_position_embeddings, 1, head_dim] -> we use [max_pos, head_dim]
    positions shape: [num_tokens]
    """
    num_tokens = query.shape[0]
    num_heads = query.shape[1]
    head_dim = query.shape[2]
    rotary_dim_half = head_dim // 2

    # cos_sin_cache has shape [max_pos, 1, head_dim], squeeze the middle dim
    cos_sin = cos_sin_cache.squeeze(1)  # [max_pos, head_dim]
    cos = cos_sin[:, :rotary_dim_half]  # [max_pos, rotary_dim_half]
    sin = cos_sin[:, rotary_dim_half:]  # [max_pos, rotary_dim_half]

    # Cast to float32 for computation
    orig_q_dtype = query.dtype
    orig_k_dtype = key.dtype
    q = query.float()
    k = key.float()
    cos_f = cos.float()
    sin_f = sin.float()

    BLOCK_SIZE = triton.next_power_of_2(rotary_dim_half)

    grid = (num_tokens, num_heads)

    _rotary_embedding_kernel[grid](
        q,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        cos_f,
        sin_f,
        cos_f.stride(0),
        sin_f.stride(0),
        positions,
        ROTARY_DIM_HALF=rotary_dim_half,
        BLOCK_SIZE=BLOCK_SIZE,
        HEAD_DIM=head_dim,
        num_warps=4,
    )

    return q.to(orig_q_dtype), k.to(orig_k_dtype)
