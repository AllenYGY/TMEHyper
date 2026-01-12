#!/usr/bin/env python3
"""
SP-HyperRAE: Spatial Perturbation Prediction via Hypergraph-Regularized Autoencoder

基于超图正则化自编码器的空间扰动预测

核心创新:
1. A1: 谱超图卷积 - 频率分解 (h_low: 全局模式, h_high: 局部变化)
2. B1: 超边内聚合损失 - 同超边内节点表示相近
3. B2: 超边类型对比损失 - 同类型超边相似，不同类型区分
4. B3: 超边间分离损失 - 防止表示坍塌
5. C1: TME嵌入变化预测 - 可解释性输出

使用方法:
    from sp_hyperrae import SPHyperRAE, ModelConfig
    from sp_hyperrae.data import load_and_preprocess_data
    from sp_hyperrae.train import train_model, evaluate_model
"""

from .config import Config, LossConfig, ModelConfig, SpectralConfig
from .contrastive import HyperedgeContrastiveLoss, HyperedgeEmbeddingExtractor
from .data import SpatialPerturbDataset, create_dataloaders, load_and_preprocess_data
from .decoder import MultiTaskDecoder, RAEDecoder
from .encoders import FallbackCellEncoder, ScGPTCellEncoder, ScGPTGuidedFusion
from .hypergraph import build_semantic_hyperedges, build_spatial_graph
from .losses import SimpleLoss, SPHyperRAELoss
from .model import SPHyperRAE
from .spectral_conv import (
    HypergraphLaplacian,
    MultiLayerSpectralHypergraphEncoder,
    SpectralHypergraphConv,
)
from .train import evaluate_model, train_model

__version__ = "2.0.0"
__author__ = "SP-HyperRAE Team"

__all__ = [
    # 配置
    "Config",
    "LossConfig",
    "SpectralConfig",
    "ModelConfig",
    # 模型
    "SPHyperRAE",
    # 数据
    "load_and_preprocess_data",
    "create_dataloaders",
    "SpatialPerturbDataset",
    # 超图
    "build_spatial_graph",
    "build_semantic_hyperedges",
    # 训练
    "train_model",
    "evaluate_model",
    # 损失
    "SPHyperRAELoss",
    "SimpleLoss",
    # 谱卷积
    "HypergraphLaplacian",
    "SpectralHypergraphConv",
    "MultiLayerSpectralHypergraphEncoder",
    # 对比学习
    "HyperedgeContrastiveLoss",
    "HyperedgeEmbeddingExtractor",
    # 解码器
    "RAEDecoder",
    "MultiTaskDecoder",
    # 编码器
    "ScGPTCellEncoder",
    "FallbackCellEncoder",
    "ScGPTGuidedFusion",
]
