"""Periodic FNO implemented directly with PyTorch real FFTs."""
import torch
from torch import nn
from torch.nn import functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, width, modes=12):
        super().__init__()
        self.modes = modes
        scale = 1 / width
        self.positive = nn.Parameter(scale * torch.randn(width, width, modes, modes,
                                                        dtype=torch.cfloat))
        self.negative = nn.Parameter(scale * torch.randn(width, width, modes, modes,
                                                        dtype=torch.cfloat))

    def forward(self, x):
        n, m = x.shape[-2:]
        k = self.modes
        if 2*k > n or k > m//2+1:
            raise ValueError("Retained Fourier blocks overlap or exceed the grid")
        spectrum = torch.fft.rfft2(x, norm='ortho')
        output = torch.zeros_like(spectrum)
        output[:, :, :k, :k] = torch.einsum('bixy,ioxy->boxy',
                                          spectrum[:, :, :k, :k], self.positive)
        output[:, :, -k:, :k] = torch.einsum('bixy,ioxy->boxy',
                                           spectrum[:, :, -k:, :k], self.negative)
        return torch.fft.irfft2(output, s=(n, m), norm='ortho')


class FNO2d(nn.Module):
    def __init__(self, modes=12, width=32, blocks=4, projection_width=64):
        super().__init__()
        self.config = dict(modes=modes, width=width, blocks=blocks,
                           projection_width=projection_width)
        self.lift = nn.Conv2d(4, width, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, modes) for _ in range(blocks)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(blocks)])
        self.project = nn.Sequential(nn.Conv2d(width, projection_width, 1), nn.GELU(),
                                     nn.Conv2d(projection_width, 1, 1))

    def forward(self, x):
        x = self.lift(x)
        for spectral, local in zip(self.spectral, self.local):
            x = F.gelu(spectral(x) + local(x))
        return self.project(x)


def parameter_counts(model):
    parameters = [p for p in model.parameters() if p.requires_grad]
    return {'trainable_parameter_elements': sum(p.numel() for p in parameters),
            'trainable_real_scalar_parameters': sum(p.numel() * (2 if p.is_complex() else 1)
                                                    for p in parameters)}
