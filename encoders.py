#!/usr/bin/env python3
"""
编码器模块

包含:
1. ScGPTCellEncoder - 基于scGPT的细胞编码器
2. FallbackCellEncoder - 备用MLP编码器
3. ScGPTGuidedFusion - scGPT引导融合模块
"""

import os
import sys

import torch
import torch.nn as nn

# 尝试导入scGPT
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scGPT"))

try:
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab

    SCGPT_AVAILABLE = True
except ImportError:
    SCGPT_AVAILABLE = False
    TransformerModel = None
    GeneVocab = None


class ScGPTCellEncoder(nn.Module):
    """
    使用scGPT作为细胞表达编码器

    scGPT架构:
    - GeneEncoder: 基因token嵌入
    - ValueEncoder: 表达值编码 (连续值)
    - TransformerEncoder: 多层Transformer
    - 输出: cell_emb (CLS token的表示)
    """

    def __init__(self, config, vocab=None, pretrained_model=None):
        super().__init__()

        self.config = config
        self.vocab = vocab
        self.d_model = config.scgpt_embed_dim

        if pretrained_model is not None:
            self.scgpt = pretrained_model
            print(f"使用预训练scGPT, embed_dim={self.d_model}")
        else:
            print("创建新的scGPT模型 (无预训练权重)")
            self.scgpt = self._create_scgpt_model(config, vocab)

        # 投影到统一维度
        self.proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, config.latent_dim),
            nn.GELU(),
            nn.LayerNorm(config.latent_dim),
        )

        # 备用编码器
        self.fallback_encoder = nn.Sequential(
            nn.Linear(config.n_genes, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        )

    def _create_scgpt_model(self, config, vocab):
        """创建scGPT模型"""
        if vocab is None:
            raise ValueError("vocab is required to create scGPT model")

        model = TransformerModel(
            ntoken=len(vocab),
            d_model=config.scgpt_embed_dim,
            nhead=8,
            d_hid=config.scgpt_embed_dim * 4,
            nlayers=6,
            vocab=vocab,
            dropout=config.dropout,
            pad_token="<pad>",
            pad_value=0,
            do_mvc=False,
            do_dab=False,
            use_batch_labels=False,
            input_emb_style="continuous",
            cell_emb_style="cls",
        )
        return model

    def forward(self, x, gene_ids=None, values=None, src_key_padding_mask=None):
        """
        前向传播

        Args:
            x: [batch, n_genes] 原始表达矩阵
            gene_ids: [batch, seq_len] 基因token ID (可选)
            values: [batch, seq_len] 表达值 (可选)
            src_key_padding_mask: [batch, seq_len] padding mask (可选)

        Returns:
            cell_emb: [batch, latent_dim] 细胞嵌入
        """
        if gene_ids is not None and values is not None:
            transformer_output = self.scgpt._encode(
                gene_ids, values, src_key_padding_mask
            )
            cell_emb = transformer_output[:, 0, :]
        else:
            cell_emb = self.fallback_encoder(x)

        cell_emb = self.proj(cell_emb)
        return cell_emb


class FallbackCellEncoder(nn.Module):
    """
    备用编码器 (当scGPT不可用时使用)
    简单的MLP + Transformer编码器
    """

    def __init__(self, n_genes, config):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(n_genes, config.scgpt_embed_dim),
            nn.LayerNorm(config.scgpt_embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.scgpt_embed_dim,
            nhead=8,
            dim_feedforward=config.scgpt_embed_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.output_proj = nn.Sequential(
            nn.LayerNorm(config.scgpt_embed_dim),
            nn.Linear(config.scgpt_embed_dim, config.latent_dim),
            nn.GELU(),
            nn.LayerNorm(config.latent_dim),
        )

    def forward(self, x):
        """
        Args:
            x: [batch, n_genes] 表达矩阵
        Returns:
            cell_emb: [batch, latent_dim]
        """
        h = self.input_proj(x)
        h = h.unsqueeze(1)
        h = self.transformer(h)
        h = h.squeeze(1)
        return self.output_proj(h)


class ScGPTGuidedFusion(nn.Module):
    """
    使用scGPT的大尺度表示引导其他通道

    核心思想:
    - scGPT包含丰富的生物学先验知识
    - 通过跨注意力让TME和扰动表示与scGPT对齐
    - 使小通道(32维)获得大模型(512维)的引导
    """

    def __init__(self, scgpt_dim, small_dim, n_heads=4, dropout=0.1):
        """
        Args:
            scgpt_dim: scGPT嵌入维度 (512)
            small_dim: 小通道维度 (32)
            n_heads: 注意力头数
        """
        super().__init__()

        self.scgpt_dim = scgpt_dim
        self.small_dim = small_dim

        # 将scGPT投影到多个"知识槽位"供查询
        self.scgpt_to_kv = nn.Sequential(
            nn.Linear(scgpt_dim, scgpt_dim),
            nn.GELU(),
            nn.Linear(scgpt_dim, small_dim * 2),
        )

        # 小通道作为Query
        self.query_proj = nn.Linear(small_dim, small_dim)

        # 跨注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=small_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )

        # 门控融合
        self.gate = nn.Sequential(nn.Linear(small_dim * 2, small_dim), nn.Sigmoid())

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.LayerNorm(small_dim),
            nn.Linear(small_dim, small_dim),
        )

    def forward(self, scgpt_emb, small_emb):
        """
        Args:
            scgpt_emb: [batch, scgpt_dim] scGPT细胞嵌入
            small_emb: [batch, small_dim] 小通道嵌入

        Returns:
            guided_emb: [batch, small_dim] 引导后的嵌入
        """
        # scGPT -> K, V
        kv = self.scgpt_to_kv(scgpt_emb)
        k, v = kv.chunk(2, dim=-1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

        # 小通道作为Query
        q = self.query_proj(small_emb).unsqueeze(1)

        # 跨注意力
        attn_out, _ = self.cross_attn(q, k, v)
        attn_out = attn_out.squeeze(1)

        # 门控融合
        gate = self.gate(torch.cat([small_emb, attn_out], dim=-1))
        guided = small_emb + gate * attn_out

        return self.output_proj(guided)


class TMEPredictor(nn.Module):
    """
    TME嵌入变化预测器 (C1)

    预测扰动后的TME嵌入变化，用于可解释性分析
    """

    def __init__(self, config):
        """
        Args:
            config: 配置对象
        """
        super().__init__()

        # 输入: z_cell + z_tme + z_perturb
        input_dim = config.latent_dim + config.tme_embed_dim + config.perturb_embed_dim

        self.predictor = nn.Sequential(
            nn.Linear(input_dim, config.tme_embed_dim * 2),
            nn.LayerNorm(config.tme_embed_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.tme_embed_dim * 2, config.tme_embed_dim),
        )

    def forward(self, z_cell, z_tme, z_perturb):
        """
        预测TME嵌入变化

        Args:
            z_cell: [batch, latent_dim] 细胞表示
            z_tme: [batch, tme_embed_dim] TME表示
            z_perturb: [batch, perturb_embed_dim] 扰动表示

        Returns:
            z_tme_post: [batch, tme_embed_dim] 预测的扰动后TME嵌入
        """
        # 拼接输入
        combined = torch.cat([z_cell, z_tme, z_perturb], dim=-1)

        # 预测变化量
        delta_z_tme = self.predictor(combined)

        # 扰动后TME = 扰动前 + 变化量
        z_tme_post = z_tme + delta_z_tme

        return z_tme_post
