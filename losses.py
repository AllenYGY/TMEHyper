#!/usr/bin/env python3
"""
损失函数模块

损失函数体系:
┌─────────────────────────────────────────────────────────────────┐
│  L_total = L_recon + α₁L_intra + α₂L_type + α₃L_inter          │
│            + α₄L_TME + α₅L_freq + α₆L_KL                       │
├─────────────────────────────────────────────────────────────────┤
│  主任务损失:                                                     │
│    - L_recon: 扰动基因预测 (MSE)                                │
│                                                                  │
│  结构约束损失 (为主任务服务):                                    │
│    - L_intra (B1): 超边内聚合                                    │
│    - L_type (B2): 类型对比                                       │
│    - L_inter (B3): 超边分离                                      │
│    - L_freq (A1): 频率正交                                       │
│    - L_TME: 结构感知多样性                                       │
│                                                                  │
│  正则化损失:                                                     │
│    - L_KL: VAE KL散度                                           │
└─────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Set

from .contrastive import HyperedgeContrastiveLoss
from .spectral_conv import compute_frequency_orthogonality_loss


class ReconstructionLoss(nn.Module):
    """重建损失（主任务）"""

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred: [batch, n_genes] 预测的表达变化
            target: [batch, n_genes] 真实的表达变化

        Returns:
            loss: MSE损失
        """
        return F.mse_loss(pred, target, reduction=self.reduction)


class KLDivergenceLoss(nn.Module):
    """VAE KL散度损失"""

    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        """
        Args:
            mu: [batch, dim] 均值
            logvar: [batch, dim] 对数方差

        Returns:
            loss: KL散度
        """
        if mu is None or logvar is None:
            return torch.tensor(0.0)

        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return kl


class TMEDiversityLoss(nn.Module):
    """
    TME多样性损失

    鼓励不同细胞的TME表示具有多样性
    通过最小化非对角元素的相似度实现
    """

    def __init__(self):
        super().__init__()

    def forward(self, z_tme):
        """
        Args:
            z_tme: [batch, dim] TME表示

        Returns:
            loss: 多样性损失
        """
        if z_tme is None or z_tme.size(0) < 2:
            return torch.tensor(0.0, device=z_tme.device if z_tme is not None else 'cpu')

        # 归一化
        z_norm = F.normalize(z_tme, dim=-1)

        # 计算相似度矩阵
        sim = torch.mm(z_norm, z_norm.t())

        # 取非对角元素
        mask = torch.eye(sim.size(0), device=sim.device).bool()
        off_diag = sim[~mask]

        # 最小化非对角元素的平方（鼓励多样性）
        loss = off_diag.pow(2).mean()

        return loss


class SPHyperRAELoss(nn.Module):
    """
    SP-HyperRAE 完整损失函数

    整合所有损失项并按权重加权
    """

    def __init__(self, config):
        """
        Args:
            config: 配置对象，包含所有损失权重
        """
        super().__init__()

        self.config = config

        # 各损失模块
        self.recon_loss = ReconstructionLoss()
        self.kl_loss = KLDivergenceLoss()
        self.tme_diversity_loss = TMEDiversityLoss()
        self.hyperedge_contrast_loss = HyperedgeContrastiveLoss(config)

        # 权重
        self.recon_weight = config.recon_weight
        self.kl_weight = config.kl_weight
        self.tme_diversity_weight = config.tme_diversity_weight
        self.freq_weight = config.freq_weight
        self.noise_weight = getattr(config, "noise_weight", 0.0)
        self.noise_logvar_target = getattr(config, "noise_logvar_target", -2.0)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mu: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
        z_tme: Optional[torch.Tensor] = None,
        h_low: Optional[torch.Tensor] = None,
        h_high: Optional[torch.Tensor] = None,
        z_nodes: Optional[torch.Tensor] = None,
        hyperedge_dict: Optional[Dict[str, List[Set[int]]]] = None,
        noise_logvar: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算所有损失

        Args:
            pred: [batch, n_genes] 预测
            target: [batch, n_genes] 目标
            mu: VAE均值
            logvar: VAE对数方差
            z_tme: TME表示
            h_low: 低频特征 (A1)
            h_high: 高频特征 (A1)
            z_nodes: 节点表示 (用于超边对比)
            hyperedge_dict: 超边字典 (用于超边对比)

        Returns:
            losses: {
                'recon': 重建损失,
                'kl': KL散度,
                'tme_diversity': TME多样性,
                'freq': 频率正交 (A1),
                'intra': 超边内聚合 (B1),
                'type': 类型对比 (B2),
                'inter': 超边分离 (B3),
                'total': 加权总损失
            }
        """
        device = pred.device
        losses = {}

        # ============ 主任务损失 ============
        losses['recon'] = self.recon_loss(pred, target)

        # ============ VAE正则化 ============
        if mu is not None and logvar is not None and self.kl_weight > 0:
            losses['kl'] = self.kl_loss(mu, logvar)
        else:
            losses['kl'] = torch.tensor(0.0, device=device)

        # ============ TME多样性损失 ============
        if z_tme is not None and self.tme_diversity_weight > 0:
            losses['tme_diversity'] = self.tme_diversity_loss(z_tme)
        else:
            losses['tme_diversity'] = torch.tensor(0.0, device=device)

        # ============ 频率正交损失 (A1) ============
        if h_low is not None and h_high is not None and self.freq_weight > 0:
            losses['freq'] = compute_frequency_orthogonality_loss(h_low, h_high)
        else:
            losses['freq'] = torch.tensor(0.0, device=device)

        # ============ 超边对比损失 (B1+B2+B3) ============
        if z_nodes is not None and hyperedge_dict is not None:
            contrast_losses = self.hyperedge_contrast_loss(z_nodes, hyperedge_dict)
            losses['intra'] = contrast_losses['intra']
            losses['type'] = contrast_losses['type']
            losses['inter'] = contrast_losses['inter']
        else:
            losses['intra'] = torch.tensor(0.0, device=device)
            losses['type'] = torch.tensor(0.0, device=device)
            losses['inter'] = torch.tensor(0.0, device=device)

        # ============ 噪声正则 ============
        if noise_logvar is not None and self.noise_weight > 0:
            target = torch.full_like(noise_logvar, self.noise_logvar_target)
            losses['noise'] = F.mse_loss(noise_logvar, target)
        else:
            losses['noise'] = torch.tensor(0.0, device=device)

        # ============ 加权总损失 ============
        total = (
            self.recon_weight * losses['recon'] +
            self.kl_weight * losses['kl'] +
            self.tme_diversity_weight * losses['tme_diversity'] +
            self.freq_weight * losses['freq'] +
            self.config.intra_weight * losses['intra'] +
            self.config.type_weight * losses['type'] +
            self.config.inter_weight * losses['inter'] +
            self.noise_weight * losses['noise']
        )

        losses['total'] = total

        return losses


class SimpleLoss(nn.Module):
    """
    简化版损失函数

    只包含重建损失和基础正则化，用于快速实验
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.recon_loss = ReconstructionLoss()
        self.kl_loss = KLDivergenceLoss()
        self.tme_diversity_loss = TMEDiversityLoss()
        self.noise_weight = getattr(config, "noise_weight", 0.0)
        self.noise_logvar_target = getattr(config, "noise_logvar_target", -2.0)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mu: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
        z_tme: Optional[torch.Tensor] = None,
        noise_logvar: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算简化损失

        Args:
            pred: 预测
            target: 目标
            mu, logvar: VAE参数
            z_tme: TME表示

        Returns:
            losses: 损失字典
        """
        device = pred.device
        losses = {}

        # 重建损失
        losses['recon'] = self.recon_loss(pred, target)

        # KL损失
        if mu is not None and logvar is not None and self.config.kl_weight > 0:
            losses['kl'] = self.kl_loss(mu, logvar)
        else:
            losses['kl'] = torch.tensor(0.0, device=device)

        # TME多样性
        if z_tme is not None and self.config.tme_diversity_weight > 0:
            losses['tme_diversity'] = self.tme_diversity_loss(z_tme)
        else:
            losses['tme_diversity'] = torch.tensor(0.0, device=device)

        # 噪声正则
        if noise_logvar is not None and self.noise_weight > 0:
            target = torch.full_like(noise_logvar, self.noise_logvar_target)
            losses['noise'] = F.mse_loss(noise_logvar, target)
        else:
            losses['noise'] = torch.tensor(0.0, device=device)

        # 总损失
        total = (
            self.config.recon_weight * losses['recon'] +
            self.config.kl_weight * losses['kl'] +
            self.config.tme_diversity_weight * losses['tme_diversity'] +
            self.noise_weight * losses['noise']
        )

        losses['total'] = total

        return losses
