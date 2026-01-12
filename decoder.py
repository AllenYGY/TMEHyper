#!/usr/bin/env python3
"""
RAE解码器模块

核心创新:
1. 条件输入: [z_cell, z_tme, z_perturb]
2. FiLM调制: TME通过仿射变换调制生成过程
3. 残差MLP: 深度网络保证表达能力

FiLM (Feature-wise Linear Modulation):
h = γ * norm(x) + β + x
其中γ和β由TME表示生成
"""

import torch
import torch.nn as nn


class ResidualDecoderBlock(nn.Module):
    """带FiLM调制的残差解码块"""

    def __init__(self, dim, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, gamma, beta):
        """
        FiLM调制的残差前向传播

        h = γ * norm(x) + β + x
        """
        # FiLM调制
        h = self.norm1(x)
        h = gamma * h + beta

        # 前馈 + 残差
        h = x + self.ff(h)
        h = self.norm2(h)

        return h


class RAEDecoder(nn.Module):
    """
    RAE解码器 - TME条件生成

    架构:
    ┌─────────────────────────────────────────┐
    │  输入: [z_cell, z_tme, z_perturb]       │
    │         (512)  (32)    (32) = 576      │
    └────────────────┬────────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────────────┐
    │  FiLM Generator (from z_tme)            │
    │  → γ, β for each layer                  │
    └────────────────┬────────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────────────┐
    │  Input Projection                        │
    │  576 → 512                              │
    └────────────────┬────────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────────────┐
    │  Residual Decoder Blocks × N            │
    │  with FiLM modulation                   │
    └────────────────┬────────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────────────┐
    │  Output Projection                       │
    │  512 → n_genes                          │
    └────────────────┬────────────────────────┘
                     │
                     ▼
                 Δx_pred
    """

    def __init__(self, n_genes, n_perturbations, config):
        super().__init__()

        self.n_genes = n_genes
        self.latent_dim = config.latent_dim

        # 扰动嵌入
        self.perturb_embed = nn.Sequential(
            nn.Embedding(n_perturbations, config.perturb_embed_dim),
            nn.LayerNorm(config.perturb_embed_dim),
        )

        # 条件融合维度
        cond_dim = config.latent_dim + config.tme_embed_dim + config.perturb_embed_dim

        # FiLM层: TME生成调制参数
        self.film_generator = nn.Sequential(
            nn.Linear(config.tme_embed_dim, config.decoder_hidden_dim * 2),
            nn.GELU(),
            nn.Linear(config.decoder_hidden_dim * 2, config.decoder_hidden_dim * 2),
        )

        # 解码器输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(cond_dim, config.decoder_hidden_dim),
            nn.LayerNorm(config.decoder_hidden_dim),
            nn.GELU(),
        )

        # 残差解码块
        self.decoder_blocks = nn.ModuleList()
        for _ in range(config.n_decoder_layers):
            self.decoder_blocks.append(
                ResidualDecoderBlock(config.decoder_hidden_dim, config.dropout)
            )

        # 输出层
        self.output_proj = nn.Sequential(
            nn.LayerNorm(config.decoder_hidden_dim),
            nn.Linear(config.decoder_hidden_dim, config.decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(config.decoder_hidden_dim, n_genes),
        )

    def forward(self, z_cell, z_tme, perturb_idx, z_perturb_guided=None):
        """
        TME条件解码

        Args:
            z_cell: [batch, latent_dim] 细胞潜空间表示
            z_tme: [batch, tme_embed_dim] TME表示 (scGPT引导后)
            perturb_idx: [batch] 扰动类型索引
            z_perturb_guided: [batch, perturb_embed_dim] scGPT引导后的扰动表示 (可选)

        Returns:
            delta_pred: [batch, n_genes] 预测的表达变化
        """
        # 扰动嵌入
        if z_perturb_guided is not None:
            z_perturb = z_perturb_guided
        else:
            z_perturb = self.perturb_embed(perturb_idx)
            if z_perturb.dim() == 3:
                z_perturb = z_perturb.squeeze(1)

        # FiLM参数: γ和β
        film_params = self.film_generator(z_tme)
        gamma, beta = film_params.chunk(2, dim=-1)

        # 拼接条件
        cond = torch.cat([z_cell, z_tme, z_perturb], dim=-1)
        h = self.input_proj(cond)

        # FiLM调制 + 残差解码
        for block in self.decoder_blocks:
            h = block(h, gamma, beta)

        # 输出
        delta_pred = self.output_proj(h)

        return delta_pred


class MultiTaskDecoder(nn.Module):
    """
    多任务解码器

    同时输出:
    1. delta_x_pred: 扰动基因预测（主任务）
    2. z_tme_post: 预测的扰动后TME嵌入（C1辅助输出）
    """

    def __init__(self, n_genes, n_perturbations, config):
        super().__init__()

        self.main_decoder = RAEDecoder(n_genes, n_perturbations, config)

        # TME变化预测器 (C1)
        input_dim = config.latent_dim + config.tme_embed_dim + config.perturb_embed_dim
        self.tme_predictor = nn.Sequential(
            nn.Linear(input_dim, config.tme_embed_dim * 2),
            nn.LayerNorm(config.tme_embed_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.tme_embed_dim * 2, config.tme_embed_dim),
        )

        self.config = config

    def forward(self, z_cell, z_tme, perturb_idx, z_perturb_guided=None):
        """
        多任务解码

        Args:
            z_cell: [batch, latent_dim] 细胞表示
            z_tme: [batch, tme_embed_dim] TME表示
            perturb_idx: [batch] 扰动索引
            z_perturb_guided: [batch, perturb_embed_dim] 引导后扰动表示

        Returns:
            outputs: {
                'delta_x_pred': [batch, n_genes] 扰动预测,
                'z_tme_post': [batch, tme_embed_dim] 扰动后TME嵌入
            }
        """
        # 主任务：扰动预测
        delta_x_pred = self.main_decoder(z_cell, z_tme, perturb_idx, z_perturb_guided)

        # 获取扰动表示
        if z_perturb_guided is not None:
            z_perturb = z_perturb_guided
        else:
            z_perturb = self.main_decoder.perturb_embed(perturb_idx)
            if z_perturb.dim() == 3:
                z_perturb = z_perturb.squeeze(1)

        # C1: TME变化预测
        combined = torch.cat([z_cell, z_tme, z_perturb], dim=-1)
        delta_z_tme = self.tme_predictor(combined)
        z_tme_post = z_tme + delta_z_tme

        return {
            "delta_x_pred": delta_x_pred,
            "z_tme_post": z_tme_post,
        }
