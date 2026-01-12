#!/usr/bin/env python3
"""
SP-HyperRAE 配置文件

包含所有模型、训练、损失函数的配置参数
"""

import os

import torch


class Config:
    """基础配置类"""

    # ============ 路径配置 ============
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(data_dir, "sp_hyperrae_results")
    data_file = "GSE193460.h5ad"
    scgpt_model_dir = os.path.join(data_dir, "scGPT/save/scGPT_human")

    # ============ 数据参数 ============
    n_genes = 500
    k_neighbors = 15
    k_hop = 2
    max_seq_len = 1200  # scGPT最大序列长度

    # ============ 超边参数 ============
    hyperedge_eps = 2000  # DBSCAN聚类半径
    hyperedge_min_samples = 5  # DBSCAN最小样本数
    n_hypergraph_layers = 3  # 超图消息传递层数
    use_true_hypergraph = True  # 使用真正的超图编码器
    n_hyperedge_types = 4  # 超边类型数量

    # ============ 模型参数 ============
    scgpt_embed_dim = 512  # scGPT嵌入维度
    tme_hidden_dim = 32  # TME编码器隐藏维度
    tme_embed_dim = 32  # TME嵌入维度
    perturb_embed_dim = 32  # 扰动嵌入维度
    decoder_hidden_dim = 512  # 解码器隐藏维度
    latent_dim = 512  # RAE潜空间维度
    n_decoder_layers = 4  # 解码器层数
    n_gnn_layers = 3  # GNN层数
    dropout = 0.1

    # ============ 训练参数 ============
    batch_size = 32
    epochs = 50
    lr = 1e-4
    weight_decay = 0.01
    warmup_epochs = 5
    freeze_scgpt = True
    scgpt_finetune_layers = 0
    scgpt_lr_scale = 0.1
    seed = 42

    # ============ 设备 ============
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


class LossConfig:
    """损失函数配置"""

    # ============ 主任务 ============
    recon_weight = 1.0  # 扰动基因预测（核心）

    # ============ 结构约束（为主任务服务）============
    # 超边结构
    intra_weight = 0.1  # B1: 超边内聚合
    type_weight = 0.05  # B2: 类型对比
    inter_weight = 0.01  # B3: 超边分离
    inter_margin = 0.5  # B3: margin参数

    # TME表示
    tme_diversity_weight = 0.1  # 结构感知多样性

    # 频率分解
    freq_weight = 0.05  # A1: 频率正交

    # ============ VAE正则（可选）============
    kl_weight = 0.0001

    # ============ 对比学习参数 ============
    contrast_temperature = 0.1  # B2温度参数


class SpectralConfig:
    """谱超图卷积配置 (A1)"""

    use_spectral_conv = True
    chebyshev_k = 3  # 切比雪夫多项式阶数
    sigma_low = 0.5  # 低通滤波器参数
    sigma_high = 1.5  # 高通滤波器参数
    freq_dim = 16  # 频率特征维度


class ModelConfig(Config, LossConfig, SpectralConfig):
    """完整模型配置（合并所有配置）"""

    # ============ 输出控制 ============
    output_tme_features = True  # 是否输出TME特征用于分析
    output_freq_features = True  # 是否输出频率特征

    def __init__(self, **kwargs):
        """允许通过关键字参数覆盖默认配置"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self):
        """导出配置为字典"""
        return {
            k: v
            for k, v in self.__class__.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }

    def __repr__(self):
        return f"ModelConfig({self.to_dict()})"
