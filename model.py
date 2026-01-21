#!/usr/bin/env python3
"""
SP-HyperRAE 完整模型

Spatial Perturbation Prediction via Hypergraph-Regularized Autoencoder

架构:
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SP-HyperRAE 完整架构                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  输入                                                                        │
│  ────                                                                        │
│  X ∈ R^{N×G}      表达矩阵                                                   │
│  S ∈ R^{N×2}      空间坐标                                                   │
│  T ∈ {1,...,K}^N  细胞类型                                                   │
│  P ∈ {1,...,M}^N  扰动类型                                                   │
│                                                                              │
│         ┌───────────────┼───────────────┐                                   │
│         │               │               │                                   │
│         ▼               ▼               ▼                                   │
│  ┌────────────┐  ┌────────────────┐  ┌────────────┐                        │
│  │   scGPT    │  │ 谱超图编码器    │  │  扰动编码   │                        │
│  │   编码器   │  │ (A1: 频率分解) │  │            │                        │
│  │  512-dim   │  │  32-dim        │  │  32-dim    │                        │
│  └─────┬──────┘  └──────┬─────────┘  └─────┬──────┘                        │
│        │                │                  │                                │
│        │    ┌───────────┴───────────┐     │                                │
│        │    │                       │     │                                │
│        │    ▼                       ▼     │                                │
│        │  h_low               h_high      │                                │
│        │  (全局模式)          (局部变化)   │                                │
│        │    │                       │     │                                │
│        │    └───────────┬───────────┘     │                                │
│        │                │                 │                                │
│        │                ▼                 │                                │
│        │            z_tme                 │                                │
│        │    (融合后的TME嵌入)             │                                │
│        │                │                 │                                │
│        │                ▼                 │                                │
│        │    ┌─────────────────────┐       │                                │
│        │    │  超边对比学习        │       │                                │
│        │    │  (B1+B2+B3)         │       │                                │
│        │    │  结构约束损失        │       │                                │
│        │    └─────────────────────┘       │                                │
│        │                │                 │                                │
│        └────────────────┼─────────────────┘                                │
│                         │                                                   │
│                         ▼                                                   │
│         ┌──────────────────────────────────┐                               │
│         │        scGPT引导融合              │                               │
│         │   跨注意力 + 门控机制             │                               │
│         └──────────────┬───────────────────┘                               │
│                        │                                                    │
│                        ▼                                                    │
│         ┌──────────────────────────────────┐                               │
│         │           RAE解码器               │                               │
│         │          (FiLM调制)              │                               │
│         └──────────────┬───────────────────┘                               │
│                        │                                                    │
│              ┌─────────┴─────────┐                                         │
│              │                   │                                         │
│              ▼                   ▼                                         │
│          Δx_pred           z_tme_post                                      │
│       (主任务输出)         (辅助输出-C1)                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""

from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn

from .contrastive import HyperedgeEmbeddingExtractor
from .decoder import MultiTaskDecoder
from .encoders import (
    SCGPT_AVAILABLE,
    FallbackCellEncoder,
    ScGPTCellEncoder,
    ScGPTGuidedFusion,
)
from .noise_augmentation import NoiseAugmenter
from .spectral_conv import HypergraphLaplacian, MultiLayerSpectralHypergraphEncoder


class SPHyperRAE(nn.Module):
    """
    SP-HyperRAE: 基于超图正则化自编码器的空间扰动预测模型

    核心创新:
    1. A1: 谱超图卷积 - 频率分解 (h_low, h_high)
    2. B1+B2+B3: 超边对比学习 - 结构约束
    3. C1: TME嵌入变化预测 - 可解释性输出
    """

    def __init__(
        self, n_genes: int, n_perturbations: int, config, vocab=None, scgpt_model=None
    ):
        """
        Args:
            n_genes: 基因数量
            n_perturbations: 扰动类型数量
            config: 配置对象
            vocab: scGPT词汇表
            scgpt_model: 预训练scGPT模型
        """
        super().__init__()

        self.n_genes = n_genes
        self.config = config

        # ========== 细胞编码器 ==========
        if SCGPT_AVAILABLE and scgpt_model is not None:
            self.cell_encoder = ScGPTCellEncoder(config, vocab, scgpt_model)
            self.use_scgpt = True
            print("使用scGPT作为细胞编码器")
        else:
            self.cell_encoder = FallbackCellEncoder(n_genes, config)
            self.use_scgpt = False
            print("使用备用MLP编码器")

        # ========== 谱超图TME编码器 (A1) ==========
        self.use_spectral_conv = getattr(config, "use_spectral_conv", True)
        if self.use_spectral_conv:
            self.tme_encoder = MultiLayerSpectralHypergraphEncoder(
                n_genes, config, n_layers=config.n_hypergraph_layers
            )
            print("使用谱超图编码器 (A1: 频率分解)")
        else:
            # 退回到简单的MLP编码器
            self.tme_encoder = nn.Sequential(
                nn.Linear(n_genes, config.tme_hidden_dim),
                nn.LayerNorm(config.tme_hidden_dim),
                nn.GELU(),
                nn.Linear(config.tme_hidden_dim, config.tme_embed_dim),
            )
            print("使用简单MLP TME编码器")

        # ========== 扰动编码器 ==========
        self.perturb_encoder = nn.Sequential(
            nn.Embedding(n_perturbations, config.perturb_embed_dim),
            nn.LayerNorm(config.perturb_embed_dim),
        )

        # ========== scGPT引导融合 ==========
        self.tme_guided_fusion = ScGPTGuidedFusion(
            scgpt_dim=config.latent_dim,
            small_dim=config.tme_embed_dim,
            n_heads=4,
            dropout=config.dropout,
        )
        self.perturb_guided_fusion = ScGPTGuidedFusion(
            scgpt_dim=config.latent_dim,
            small_dim=config.perturb_embed_dim,
            n_heads=4,
            dropout=config.dropout,
        )

        # ========== 多任务解码器 ==========
        self.decoder = MultiTaskDecoder(n_genes, n_perturbations, config)

        # ========== 超边嵌入提取器 (用于分析) ==========
        self.hyperedge_extractor = HyperedgeEmbeddingExtractor(
            config.tme_embed_dim, config.tme_embed_dim
        )

        # ========== VAE风格的μ/σ (可选) ==========
        self.use_vae = config.kl_weight > 0
        if self.use_vae:
            self.mu_proj = nn.Linear(config.latent_dim, config.latent_dim)
            self.logvar_proj = nn.Linear(config.latent_dim, config.latent_dim)

        # ========== 噪声增强 (可选) ==========
        self.use_noise_aug = getattr(config, "use_noise_aug", False)
        if self.use_noise_aug:
            self.noise_augmenter = NoiseAugmenter(
                config.latent_dim,
                logvar_min=getattr(config, "noise_logvar_min", -6.0),
                logvar_max=getattr(config, "noise_logvar_max", 1.0),
                logvar_target=getattr(config, "noise_logvar_target", -2.0),
            )
            if self.use_vae:
                print(
                    "Warning: use_noise_aug and VAE are both enabled; noise may be excessive."
                )

        # ========== 超图拉普拉斯缓存 ==========
        self.L_tilde = None
        self.hyperedge_dict = None
        self.node_to_hyperedges = None

    def set_hypergraph(
        self,
        hyperedge_dict: Dict[str, List[Set[int]]],
        node_to_hyperedges: Dict[int, List],
        n_nodes: int,
        device: str = "cpu",
    ):
        """
        设置超图结构

        Args:
            hyperedge_dict: 超边字典
            node_to_hyperedges: 节点到超边映射
            n_nodes: 节点数量
            device: 设备
        """
        self.hyperedge_dict = hyperedge_dict
        self.node_to_hyperedges = node_to_hyperedges

        if self.use_spectral_conv:
            # 构建超图拉普拉斯
            laplacian = HypergraphLaplacian(n_nodes, hyperedge_dict, node_to_hyperedges)
            self.L_tilde = laplacian.to_torch(device)
            self.tme_encoder.set_laplacian(self.L_tilde)
            print(f"超图拉普拉斯设置完成: λ_max={laplacian.lambda_max:.2f}")

    def encode_cell(self, x, gene_ids=None, values=None, padding_mask=None):
        """编码细胞表达"""
        if self.use_scgpt:
            z = self.cell_encoder(x, gene_ids, values, padding_mask)
        else:
            z = self.cell_encoder(x)

        # VAE重参数化
        if self.use_vae:
            mu = self.mu_proj(z)
            logvar = self.logvar_proj(z)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z, mu, logvar

        return z, None, None

    def encode_tme(self, x, center_indices, scgpt_emb=None):
        """
        编码TME

        Args:
            x: [N, n_genes] 全部节点特征
            center_indices: [batch] 中心节点索引
            scgpt_emb: [batch, latent_dim] scGPT嵌入

        Returns:
            z_tme: [batch, tme_embed_dim]
            h_low: [batch, freq_dim] (仅谱卷积)
            h_high: [batch, freq_dim] (仅谱卷积)
        """
        if self.use_spectral_conv:
            z_tme, h_low, h_high = self.tme_encoder(x, center_indices, scgpt_emb)
            return z_tme, h_low, h_high
        else:
            # 简单编码器：只取中心节点
            center_x = x[center_indices]
            z_tme = self.tme_encoder(center_x)
            return z_tme, None, None

    def forward(
        self,
        x: torch.Tensor,
        perturb_idx: torch.Tensor,
        graph=None,
        gene_ids: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        noise_scale: Optional[float] = None,
        global_center_indices: Optional[torch.Tensor] = None,
        all_expression: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        完整前向传播

        Args:
            x: [batch, n_genes] 表达矩阵
            perturb_idx: [batch] 扰动索引
            graph: Batch graph (兼容旧接口)
            gene_ids: [batch, seq_len] scGPT基因ID
            values: [batch, seq_len] scGPT表达值
            padding_mask: [batch, seq_len] scGPT padding mask
            noise_scale: 噪声增强缩放系数（仅训练期生效）
            global_center_indices: [batch] 全局中心节点索引
            all_expression: [N, n_genes] 完整表达矩阵

        Returns:
            output: {
                'delta_x_pred': [batch, n_genes] 扰动预测,
                'z_cell': [batch, latent_dim] 细胞表示,
                'z_tme': [batch, tme_embed_dim] TME表示,
                'h_low': [batch, freq_dim] 低频特征 (A1),
                'h_high': [batch, freq_dim] 高频特征 (A1),
                'z_tme_post': [batch, tme_embed_dim] 扰动后TME (C1),
                'mu': VAE均值,
                'logvar': VAE对数方差,
                'noise_logvar': 噪声增强的logvar (可选),
                'hyperedge_features': 超边特征 (用于分析)
            }
        """
        device = x.device
        batch_size = x.size(0)

        # 1. 编码细胞
        z_cell, mu, logvar = self.encode_cell(x, gene_ids, values, padding_mask)

        # 1.1 噪声增强 (训练期)
        noise_logvar = None
        if self.use_noise_aug:
            if noise_scale is None:
                noise_scale = 1.0
            z_cell, noise_logvar = self.noise_augmenter(z_cell, noise_scale=noise_scale)

        # 2. 编码TME (谱超图卷积)
        if self.use_spectral_conv and all_expression is not None:
            full_x = (
                torch.FloatTensor(all_expression).to(device)
                if not isinstance(all_expression, torch.Tensor)
                else all_expression.to(device)
            )
            center_indices = (
                global_center_indices
                if global_center_indices is not None
                else torch.arange(batch_size).to(device)
            )
            z_tme_raw, h_low, h_high = self.encode_tme(full_x, center_indices, z_cell)
        else:
            # 使用简单编码
            z_tme_raw = (
                self.tme_encoder(x)
                if not self.use_spectral_conv
                else torch.zeros(batch_size, self.config.tme_embed_dim, device=device)
            )
            h_low, h_high = None, None

        # 3. 编码扰动
        z_perturb_raw = self.perturb_encoder[0](perturb_idx)
        z_perturb_raw = self.perturb_encoder[1](z_perturb_raw)

        # 4. scGPT引导融合
        z_tme = self.tme_guided_fusion(z_cell, z_tme_raw)
        z_perturb = self.perturb_guided_fusion(z_cell, z_perturb_raw)

        # 5. 多任务解码
        decoder_output = self.decoder(
            z_cell, z_tme, perturb_idx, z_perturb_guided=z_perturb
        )

        # 6. 构建输出
        output = {
            "delta_x_pred": decoder_output["delta_x_pred"],
            "z_cell": z_cell,
            "z_tme": z_tme,
            "h_low": h_low,
            "h_high": h_high,
            "z_tme_post": decoder_output["z_tme_post"],
            "mu": mu,
            "logvar": logvar,
            "noise_logvar": noise_logvar,
        }

        # 7. 提取超边特征 (用于分析)
        if self.hyperedge_dict is not None and self.config.output_tme_features:
            # 使用全局TME嵌入
            if all_expression is not None:
                full_z_tme = z_tme_raw if z_tme_raw is not None else z_tme
            else:
                full_z_tme = z_tme

            output["hyperedge_features"] = self.hyperedge_extractor(
                full_z_tme, self.hyperedge_dict
            )

        return output

    def get_tme_embeddings(self, all_expression: torch.Tensor) -> torch.Tensor:
        """
        获取所有细胞的TME嵌入

        用于下游分析和可视化

        Args:
            all_expression: [N, n_genes] 完整表达矩阵

        Returns:
            z_tme: [N, tme_embed_dim]
        """
        device = next(self.parameters()).device
        all_expression = all_expression.to(device)

        if self.use_spectral_conv:
            center_indices = torch.arange(all_expression.size(0)).to(device)
            z_tme, _, _ = self.tme_encoder(all_expression, center_indices)
        else:
            z_tme = self.tme_encoder(all_expression)

        return z_tme


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {"total": total, "trainable": trainable, "frozen": frozen}
