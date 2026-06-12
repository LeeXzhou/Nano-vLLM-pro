import torch
import triton
import triton.language as tl


@triton.jit
def _rotary_embedding_kernel(
    X_ptr,
    X_stride_0,
    X_stride_1,
    X_stride_2,
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

    # X shape: [num_tokens, num_heads, head_dim]
    # X_stride_0 = token stride, X_stride_1 = head stride, X_stride_2 = element stride
    x_base = X_ptr + token_idx * X_stride_0 + head_idx * X_stride_1
    x1_ptrs = x_base + half_offsets
    x2_ptrs = x_base + ROTARY_DIM_HALF + half_offsets
    x1 = tl.load(x1_ptrs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Apply rotation
    new_x1 = x1 * cos - x2 * sin
    new_x2 = x2 * cos + x1 * sin

    # Store in-place
    tl.store(x1_ptrs, new_x1, mask=mask)
    tl.store(x2_ptrs, new_x2, mask=mask)


def _apply_rotary_on_tensor(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE on a single tensor (Q or K).

    x shape: [num_tokens, num_heads, head_dim]
    cos/sin shape: [max_pos, rotary_dim_half]
    positions shape: [num_tokens]
    """
    num_tokens = x.shape[0]
    num_heads = x.shape[1]
    head_dim = x.shape[2]
    rotary_dim_half = head_dim // 2

    orig_dtype = x.dtype
    x_f = x.float()
    cos_f = cos.float()
    sin_f = sin.float()

    BLOCK_SIZE = triton.next_power_of_2(rotary_dim_half)
    grid = (num_tokens, num_heads)

    _rotary_embedding_kernel[grid](
        x_f,
        x_f.stride(0),
        x_f.stride(1),
        x_f.stride(2),
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

    return x_f.to(orig_dtype)


def triton_apply_rotary_emb(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Rotary Embedding using Triton.

    Applies RoPE on Q and K tensors separately, each with its own head count.
    query shape: [num_tokens, num_q_heads, head_dim]
    key shape: [num_tokens, num_kv_heads, head_dim]
    cos_sin_cache shape: [max_position_embeddings, 1, head_dim]
    positions shape: [num_tokens]
    """
    # cos_sin_cache has shape [max_pos, 1, head_dim], squeeze the middle dim
    cos_sin = cos_sin_cache.squeeze(1)  # [max_pos, head_dim]
    rotary_dim_half = query.shape[-1] // 2
    cos = cos_sin[:, :rotary_dim_half]  # [max_pos, rotary_dim_half]
    sin = cos_sin[:, rotary_dim_half:]  # [max_pos, rotary_dim_half]

    # Make cos/sin contiguous before passing to kernel
    cos = cos.contiguous()
    sin = sin.contiguous()

    q = _apply_rotary_on_tensor(query, positions, cos, sin)
    k = _apply_rotary_on_tensor(key, positions, cos, sin)

    return q, k
