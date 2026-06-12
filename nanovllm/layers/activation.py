import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.layers.triton_activation import triton_silu_and_mul


class SiluAndMul(nn.Module):

    def __init__(self, use_triton: bool = True):
        super().__init__()
        self.use_triton = use_triton

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_triton and x.is_cuda:
            return triton_silu_and_mul(x)
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
