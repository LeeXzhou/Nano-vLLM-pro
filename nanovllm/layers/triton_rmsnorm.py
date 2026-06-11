import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    X_ptr, X_stride,
    W_ptr, W_stride,
    Y_ptr, Y_stride,
    R_ptr, R_stride,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    eps,
    HAS_RESIDUAL: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N_COLS

    # Load input
    x_ptrs = X_ptr + row_idx * X_stride + col_offsets
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    if HAS_RESIDUAL:
        r_ptrs = R_ptr + row_idx * R_stride + col_offsets
        residual = tl.load(r_ptrs, mask=mask, other=0.0).to(tl.float32)
        x = x + residual

    # Compute RMS
    x_sq = x * x
    variance = tl.sum(x_sq, axis=0) / N_COLS
    rrms = tl.rsqrt(variance + eps)

    # Normalize and scale
    x_hat = x * rrms
    # Load weight
    w = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    y = x_hat * w

    # Store output
    y_ptrs = Y_ptr + row_idx * Y_stride + col_offsets
    tl.store(y_ptrs, y, mask=mask)

    if HAS_RESIDUAL:
        # Store residual (the x before normalization, casted back)
        r_out_ptrs = R_ptr + row_idx * R_stride + col_offsets
        tl.store(r_out_ptrs, x, mask=mask)


def triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fused RMSNorm using Triton (no residual add)."""
    orig_dtype = x.dtype
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).contiguous().float()
    n_rows, n_cols = x_2d.shape
    y = torch.empty_like(x_2d)
    w = weight.float()

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4 if BLOCK_SIZE < 512 else (8 if BLOCK_SIZE < 2048 else 16)

    _rms_norm_kernel[(n_rows,)](
        x_2d, x_2d.stride(0),
        w, w.stride(0),
        y, y.stride(0),
        None, 0,
        N_COLS=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        eps=eps,
        HAS_RESIDUAL=False,
        num_warps=num_warps,
    )
    return y.to(orig_dtype).reshape(orig_shape)


def triton_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused residual add + RMSNorm using Triton.

    Computes:
    residual = x.float() + residual.float() (in-place update on residual)
    y = rms_norm(residual, weight)
    Returns (y, residual.cast(orig_dtype))
    """
    orig_dtype = x.dtype
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).contiguous().float()
    residual_2d = residual.reshape(-1, residual.shape[-1]).contiguous().float()
    n_rows, n_cols = x_2d.shape
    y = torch.empty_like(x_2d)
    w = weight.float()

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4 if BLOCK_SIZE < 512 else (8 if BLOCK_SIZE < 2048 else 16)

    _rms_norm_kernel[(n_rows,)](
        x_2d, x_2d.stride(0),
        w, w.stride(0),
        y, y.stride(0),
        residual_2d, residual_2d.stride(0),
        N_COLS=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        eps=eps,
        HAS_RESIDUAL=True,
        num_warps=num_warps,
    )
    return y.to(orig_dtype).reshape(orig_shape), residual_2d.to(orig_dtype).reshape(orig_shape)
