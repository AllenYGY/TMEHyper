#!/usr/bin/env python3
"""
Noise augmentation module.

Applies learnable, per-dimension noise to latent embeddings during training.
"""

import torch
import torch.nn as nn


class NoiseAugmenter(nn.Module):
    """Learnable noise injection for latent embeddings."""

    def __init__(
        self,
        dim: int,
        logvar_min: float = -6.0,
        logvar_max: float = 1.0,
        logvar_target: float = -2.0,
    ):
        super().__init__()
        self.logvar_head = nn.Linear(dim, dim)
        nn.init.constant_(self.logvar_head.bias, logvar_target)
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

    def forward(self, z: torch.Tensor, noise_scale: float = 1.0):
        """
        Args:
            z: [batch, dim] latent embedding
            noise_scale: scalar scale for noise magnitude (warmup friendly)

        Returns:
            z_noisy: [batch, dim] noisy embedding (training only)
            logvar: [batch, dim] predicted log variance
        """
        logvar = self.logvar_head(z)
        logvar = torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)

        if self.training and noise_scale > 0:
            sigma = torch.exp(0.5 * logvar)
            eps = torch.randn_like(z)
            z_noisy = z + noise_scale * sigma * eps
        else:
            z_noisy = z

        return z_noisy, logvar
