import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.layers.triton_activation import triton_silu_and_mul


class SiluAndMul(nn.Module):

    def __init__(self, use_triton: bool = True):
        super().__init__()
        self.use_triton = use_triton

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

    def _triton_forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_silu_and_mul(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_triton and x.is_cuda:
            return self._triton_forward(x)
        return self.forward(x)
