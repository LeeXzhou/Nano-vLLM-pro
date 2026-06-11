import torch
import triton
import triton.language as tl


@triton.jit
def _silu_and_mul_kernel(
    GATE_ptr, GATE_stride,
    UP_ptr, UP_stride,
    Y_ptr, Y_stride,
    HALF: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < HALF

    # Load gate and up separately
    gate_ptrs = GATE_ptr + row_idx * GATE_stride + col_offsets
    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)

    up_ptrs = UP_ptr + row_idx * UP_stride + col_offsets
    up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)

    # SiLU(gate) * up
    # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    sigmoid_gate = gate / (1.0 + tl.exp(-gate))
    y = sigmoid_gate * up

    # Store
    y_ptrs = Y_ptr + row_idx * Y_stride + col_offsets
    tl.store(y_ptrs, y, mask=mask)


def triton_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Fused SiLU + Mul using Triton.

    Input x has shape [..., 2*hidden], output has shape [..., hidden].
    Computes: SiLU(x[..., :hidden]) * x[..., hidden:]
    """
    orig_dtype = x.dtype
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    n_rows, n_cols = x_2d.shape
    half = n_cols // 2

    gate = x_2d[:, :half].float()
    up = x_2d[:, half:].float()
    y = torch.empty(n_rows, half, dtype=torch.float32, device=x.device)

    BLOCK_SIZE = triton.next_power_of_2(half)
    num_warps = 4 if BLOCK_SIZE < 512 else (8 if BLOCK_SIZE < 2048 else 16)

    _silu_and_mul_kernel[(n_rows,)](
        gate, gate.stride(0),
        up, up.stride(0),
        y, y.stride(0),
        HALF=half,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y.to(orig_dtype).reshape(*x.shape[:-1], half)
