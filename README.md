# SP-HyperRAE

<p align="center">
  <b>Spatial Perturbation Prediction via Hypergraph-Regularized Autoencoder</b>
</p>

<p align="center">
  基于超图正则化自编码器的空间转录组扰动预测模型
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#background">Background</a> •
  <a href="#method">Method</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#experiments">Experiments</a>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Background](#background)
  - [Motivation](#motivation)
  - [Problem Formulation](#problem-formulation)
  - [Related Work](#related-work)
- [Method](#method)
  - [Architecture Overview](#architecture-overview)
  - [Semantic Hyperedge Construction](#1-semantic-hyperedge-construction)
  - [Spectral Hypergraph Convolution](#2-spectral-hypergraph-convolution)
  - [Hyperedge Contrastive Learning](#3-hyperedge-contrastive-learning)
  - [Multi-Task Decoder](#4-multi-task-decoder)
  - [Loss Function](#5-loss-function)
  - [Training Strategy](#6-training-strategy)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Format](#data-format)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Experiments](#experiments)
- [Visualization](#visualization)
- [Biological Interpretation](#biological-interpretation)
- [Extending SP-HyperRAE](#extending-sp-hyperrae)

---

## Overview

SP-HyperRAE (**S**patial **P**erturbation prediction via **Hyper**graph-**R**egularized **A**uto**E**ncoder) 是一个专为空间转录组学数据设计的深度学习模型，用于预测细胞在特定扰动条件（如药物处理、基因敲除等）下的基因表达变化。

### Key Innovations

| Innovation | Module | Description |
|------------|--------|-------------|
| **A1** | Spectral Hypergraph Convolution | 基于切比雪夫多项式的谱超图卷积，支持频率分解 |
| **B1** | Intra-hyperedge Cohesion | 超边内聚合损失，使同一微环境内细胞表示相近 |
| **B2** | Type-level Contrast | 超边类型对比损失，区分不同类型微环境 |
| **B3** | Inter-hyperedge Separation | 超边间分离损失，防止表示坍塌 |
| **C1** | TME Change Prediction | 扰动后TME嵌入预测，增强可解释性 |

### Key Features

- **语义超图构建**：基于细胞类型和空间位置构建四种生物学驱动的超边类型（T_contact, Tumor_contact, Interface, Spatial）
- **谱超图卷积**：使用切比雪夫多项式近似避免显式特征分解，支持大规模数据
- **频率分解**：分离低频（全局组织模式）和高频（局部微环境变化）特征，提供生物学可解释性
- **超边对比学习**：通过三种结构约束损失增强超图表示的质量
- **scGPT集成**：支持预训练单细胞语言模型作为细胞编码器，利用大规模预训练知识
- **多任务学习**：同时预测扰动响应和扰动后TME状态
- **灵活配置**：支持多种运行模式，从简单MLP到完整超图模型

### Why SP-HyperRAE?

传统的扰动预测方法主要基于单细胞表达数据，忽略了细胞间的空间关系。然而，在真实的组织环境中：

1. **细胞响应受微环境影响**：相邻细胞通过旁分泌信号、物理接触等方式相互影响
2. **空间结构蕴含生物学信息**：肿瘤-免疫界面、免疫浸润区等特殊区域具有独特的功能特性
3. **高阶关系普遍存在**：一个细胞可能同时受到多个邻居的影响，普通图无法有效建模

SP-HyperRAE 通过**超图**来建模这些高阶空间关系，通过**谱卷积**来聚合邻域信息，从而实现更准确的扰动预测。

---

## Background

### Motivation

空间转录组学技术（如10x Visium, MERFISH, seqFISH+等）能够同时获取细胞的基因表达谱和空间位置信息，为理解组织微环境提供了前所未有的机会。然而，现有的扰动预测方法面临以下挑战：

**挑战1：空间信息未被充分利用**

现有方法（如GEARS, CPA等）主要基于单细胞表达数据，将每个细胞视为独立样本，忽略了空间转录组数据中丰富的空间结构信息。

**挑战2：普通图无法建模高阶关系**

即使使用图神经网络，普通图的边只能连接两个节点，难以建模"一组细胞共同构成一个功能单元"这样的高阶关系。

**挑战3：微环境异质性**

肿瘤微环境高度异质，不同区域（如肿瘤核心、浸润边缘、免疫富集区）的细胞对扰动的响应可能截然不同。

### Problem Formulation

给定：

- 基因表达矩阵 $X \in \mathbb{R}^{N \times G}$，其中 $N$ 是细胞数，$G$ 是基因数
- 空间坐标 $S \in \mathbb{R}^{N \times 2}$
- 细胞类型标签 $T \in \{1, ..., K\}^N$
- 扰动标签 $P \in \{1, ..., M\}^N$
- 扰动前后的表达变化 $\Delta X \in \mathbb{R}^{N \times G}$（训练时）

目标：

- 学习一个函数 $f: (X, S, T, P) \rightarrow \Delta X$
- 对于新的细胞和扰动组合，预测其表达变化

### Related Work

#### Single-cell Perturbation Prediction

| Method | Year | Key Idea |
|--------|------|----------|
| scGen | 2019 | VAE-based, learns perturbation vector |
| CPA | 2021 | Compositional perturbation autoencoder |
| GEARS | 2022 | GNN for gene regulatory network |
| scGPT | 2023 | Transformer pre-training for single-cell |

#### Graph/Hypergraph Neural Networks

| Method | Year | Key Idea |
|--------|------|----------|
| ChebNet | 2016 | Chebyshev polynomial approximation for spectral convolution |
| GAT | 2018 | Attention-based graph convolution |
| HGNN | 2019 | Hypergraph neural network with incidence matrix |
| HyperGCN | 2019 | Approximate hypergraph as simple graph |
| AllSet | 2022 | Set function based hypergraph learning |

#### Spatial Transcriptomics Analysis

| Method | Year | Key Idea |
|--------|------|----------|
| SpaGCN | 2021 | GCN for spatial domain identification |
| STAGATE | 2022 | Graph attention for spatial clustering |
| SpaceFlow | 2022 | Spatially-regularized deep learning |

**SP-HyperRAE 的定位**：结合空间转录组学和超图学习，专注于扰动预测任务，填补了"空间感知的扰动预测"这一研究空白。

---

## Method

### Architecture Overview

```txt
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│                        SP-HyperRAE Complete Architecture                        │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                              INPUT LAYER                                 │   │
│  ├─────────────────────────────────────────────────────────────────────────┤   │
│  │                                                                         │   │
│  │   X ∈ R^{N×G}        S ∈ R^{N×2}        T ∈ {1..K}^N      P ∈ {1..M}^N │   │
│  │   (Expression)       (Spatial)          (Cell Type)       (Perturbation)│   │
│  │                                                                         │   │
│  └───────┬─────────────────┬─────────────────┬──────────────────┬─────────┘   │
│          │                 │                 │                  │              │
│          │                 │                 │                  │              │
│          ▼                 ▼                 ▼                  ▼              │
│  ┌───────────────────────────────────────────────────────────────────────┐    │
│  │                         PREPROCESSING                                  │    │
│  ├───────────────────────────────────────────────────────────────────────┤    │
│  │                                                                        │    │
│  │   1. Gene Selection (HVG)    ──→  X' ∈ R^{N×500}                      │    │
│  │   2. Spatial KNN Graph       ──→  Edge Index, Adjacency List          │    │
│  │   3. Cell Type Mapping       ──→  T_cell(0), Tumor(1), Overlap(2)...  │    │
│  │   4. Neighborhood Features   ──→  T_ratio, Tumor_ratio, Is_interface  │    │
│  │                                                                        │    │
│  └───────────────────────────────────────────────────────────────────────┘    │
│          │                                                                     │
│          ▼                                                                     │
│  ┌───────────────────────────────────────────────────────────────────────┐    │
│  │                    HYPEREDGE CONSTRUCTION                              │    │
│  ├───────────────────────────────────────────────────────────────────────┤    │
│  │                                                                        │    │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │    │
│  │   │  T_contact   │  │Tumor_contact │  │  Interface   │  │ Spatial  │  │    │
│  │   │              │  │              │  │              │  │          │  │    │
│  │   │ T-cell ratio │  │ Tumor ratio  │  │ Has both T   │  │   KNN    │  │    │
│  │   │ stratified   │  │ stratified   │  │ and Tumor    │  │ neighbors│  │    │
│  │   │ + DBSCAN     │  │ + DBSCAN     │  │ + DBSCAN     │  │          │  │    │
│  │   └──────────────┘  └──────────────┘  └──────────────┘  └──────────┘  │    │
│  │                                                                        │    │
│  │   Output: hyperedge_dict = {type: [set(nodes), ...], ...}             │    │
│  │           node_to_hyperedges = {node: [(type, idx), ...], ...}        │    │
│  │                                                                        │    │
│  └───────────────────────────────────────────────────────────────────────┘    │
│          │                                                                     │
│          ▼                                                                     │
│  ┌───────────────────────────────────────────────────────────────────────┐    │
│  │                    HYPERGRAPH LAPLACIAN                                │    │
│  ├───────────────────────────────────────────────────────────────────────┤    │
│  │                                                                        │    │
│  │   Incidence Matrix:  H ∈ R^{N×M}  where H[i,j]=1 if node i in edge j  │    │
│  │                                                                        │    │
│  │   Node Degree:       D_v = diag(H·1_M)                                │    │
│  │   Edge Degree:       D_e = diag(H^T·1_N)                              │    │
│  │   Edge Weight:       W = I_M (identity)                               │    │
│  │                                                                        │    │
│  │   Laplacian:         L = I - D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}   │    │
│  │   Scaled:            L̃ = 2L/λ_max - I                                 │    │
│  │                                                                        │    │
│  └───────────────────────────────────────────────────────────────────────┘    │
│          │                                                                     │
│          ▼                                                                     │
│  ┌───────────────────────────────────────────────────────────────────────┐    │
│  │                         ENCODER STAGE                                  │    │
│  ├───────────────────────────────────────────────────────────────────────┤    │
│  │                                                                        │    │
│  │   ┌────────────────────────────────────────────────────────────────┐  │    │
│  │   │                     CELL ENCODER                               │  │    │
│  │   │                                                                │  │    │
│  │   │   Option A: scGPT (Pre-trained Transformer)                   │  │    │
│  │   │   ┌─────────────────────────────────────────────────────────┐ │  │    │
│  │   │   │  Gene Tokens + Expression Values                        │ │  │    │
│  │   │   │       ↓                                                 │ │  │    │
│  │   │   │  [CLS] + Gene Embeddings + Value Embeddings             │ │  │    │
│  │   │   │       ↓                                                 │ │  │    │
│  │   │   │  Transformer Layers (12 layers, 512 dim)                │ │  │    │
│  │   │   │       ↓                                                 │ │  │    │
│  │   │   │  CLS Token Embedding → z_cell ∈ R^{512}                 │ │  │    │
│  │   │   └─────────────────────────────────────────────────────────┘ │  │    │
│  │   │                                                                │  │    │
│  │   │   Option B: Fallback MLP                                      │  │    │
│  │   │   ┌─────────────────────────────────────────────────────────┐ │  │    │
│  │   │   │  X → Linear(500,1024) → LayerNorm → GELU                │ │  │    │
│  │   │   │    → Linear(1024,512) → LayerNorm → z_cell ∈ R^{512}    │ │  │    │
│  │   │   └─────────────────────────────────────────────────────────┘ │  │    │
│  │   │                                                                │  │    │
│  │   └────────────────────────────────────────────────────────────────┘  │    │
│  │                                                                        │    │
│  │   ┌────────────────────────────────────────────────────────────────┐  │    │
│  │   │              SPECTRAL HYPERGRAPH TME ENCODER (A1)              │  │    │
│  │   │                                                                │  │    │
│  │   │   Input: X_all ∈ R^{N×G}, L̃ ∈ R^{N×N}                         │  │    │
│  │   │                                                                │  │    │
│  │   │   Step 1: Input Projection                                    │  │    │
│  │   │   ┌─────────────────────────────────────────────────────────┐ │  │    │
│  │   │   │  h = Linear(G, 32) → LayerNorm → GELU                   │ │  │    │
│  │   │   └─────────────────────────────────────────────────────────┘ │  │    │
│  │   │                                                                │  │    │
│  │   │   Step 2: Multi-layer Spectral Convolution (×3 layers)        │  │    │
│  │   │   ┌─────────────────────────────────────────────────────────┐ │  │    │
│  │   │   │                                                         │ │  │    │
│  │   │   │   For each layer:                                       │ │  │    │
│  │   │   │                                                         │ │  │    │
│  │   │   │   Low-pass Filter:                                      │ │  │    │
│  │   │   │     h_low = Σ_{k=0}^{K} θ_k^{low} · T_k(L̃) · h          │ │  │    │
│  │   │   │                                                         │ │  │    │
│  │   │   │   High-pass Filter:                                     │ │  │    │
│  │   │   │     h_high = Σ_{k=0}^{K} θ_k^{high} · T_k(L̃) · h        │ │  │    │
│  │   │   │                                                         │ │  │    │
│  │   │   │   Chebyshev Recursion:                                  │ │  │    │
│  │   │   │     T_0(x) = I                                          │ │  │    │
│  │   │   │     T_1(x) = x                                          │ │  │    │
│  │   │   │     T_k(x) = 2x·T_{k-1}(x) - T_{k-2}(x)                 │ │  │    │
│  │   │   │                                                         │ │  │    │
│  │   │   │   Fusion:                                               │ │  │    │
│  │   │   │     h = Linear([h_low; h_high; h], 32) → LN → GELU     │ │  │    │
│  │   │   │                                                         │ │  │    │
│  │   │   └─────────────────────────────────────────────────────────┘ │  │    │
│  │   │                                                                │  │    │
│  │   │   Step 3: scGPT Guidance (Cross-attention)                    │  │    │
│  │   │   ┌─────────────────────────────────────────────────────────┐ │  │    │
│  │   │   │  Q = h · W_q                                            │ │  │    │
│  │   │   │  K, V = z_cell · W_kv (split)                           │ │  │    │
│  │   │   │  attn = softmax(QK^T/√d) · V                            │ │  │    │
│  │   │   │  gate = σ(Linear([h; attn]))                            │ │  │    │
│  │   │   │  h = LayerNorm(h + gate ⊙ attn)                         │ │  │    │
│  │   │   └─────────────────────────────────────────────────────────┘ │  │    │
│  │   │                                                                │  │    │
│  │   │   Output: z_tme ∈ R^{32}, h_low ∈ R^{16}, h_high ∈ R^{16}     │  │    │
│  │   │                                                                │  │    │
│  │   └────────────────────────────────────────────────────────────────┘  │    │
│  │                                                                        │    │
│  │   ┌────────────────────────────────────────────────────────────────┐  │    │
│  │   │                   PERTURBATION ENCODER                         │  │    │
│  │   │                                                                │  │    │
│  │   │   P → Embedding(M, 32) → LayerNorm → z_perturb ∈ R^{32}       │  │    │
│  │   │                                                                │  │    │
│  │   └────────────────────────────────────────────────────────────────┘  │    │
│  │                                                                        │    │
│  └───────────────────────────────────────────────────────────────────────┘    │
│          │                                                                     │
│          ▼                                                                     │
│  ┌───────────────────────────────────────────────────────────────────────┐    │
│  │                        FUSION STAGE                                    │    │
│  ├───────────────────────────────────────────────────────────────────────┤    │
│  │                                                                        │    │
│  │   scGPT-Guided Fusion for TME:                                        │    │
│  │   ┌─────────────────────────────────────────────────────────────────┐ │    │
│  │   │  z_tme_fused = CrossAttn(Q=z_tme, K=z_cell, V=z_cell)          │ │    │
│  │   │              + GatedResidual(z_tme)                             │ │    │
│  │   └─────────────────────────────────────────────────────────────────┘ │    │
│  │                                                                        │    │
│  │   scGPT-Guided Fusion for Perturbation:                               │    │
│  │   ┌─────────────────────────────────────────────────────────────────┐ │    │
│  │   │  z_perturb_fused = CrossAttn(Q=z_perturb, K=z_cell, V=z_cell)  │ │    │
│  │   │                  + GatedResidual(z_perturb)                     │ │    │
│  │   └─────────────────────────────────────────────────────────────────┘ │    │
│  │                                                                        │    │
│  └───────────────────────────────────────────────────────────────────────┘    │
│          │                                                                     │
│          ▼                                                                     │
│  ┌───────────────────────────────────────────────────────────────────────┐    │
│  │                        DECODER STAGE                                   │    │
│  ├───────────────────────────────────────────────────────────────────────┤    │
│  │                                                                        │    │
│  │   Multi-Task RAE Decoder with FiLM Modulation                         │    │
│  │                                                                        │    │
│  │   Input: [z_cell; z_tme_fused; z_perturb_fused] ∈ R^{576}            │    │
│  │                                                                        │    │
│  │   ┌─────────────────────────────────────────────────────────────────┐ │    │
│  │   │                    TME Conditioner                              │ │    │
│  │   │                                                                 │ │    │
│  │   │   γ = Linear(z_tme_fused, hidden_dim)   # Scale               │ │    │
│  │   │   β = Linear(z_tme_fused, hidden_dim)   # Shift               │ │    │
│  │   │                                                                 │ │    │
│  │   └─────────────────────────────────────────────────────────────────┘ │    │
│  │                                                                        │    │
│  │   ┌─────────────────────────────────────────────────────────────────┐ │    │
│  │   │              Residual Decoder Blocks (×4)                       │ │    │
│  │   │                                                                 │ │    │
│  │   │   For each block:                                               │ │    │
│  │   │     h' = Linear(h) → LayerNorm                                  │ │    │
│  │   │     h' = γ ⊙ h' + β              # FiLM modulation             │ │    │
│  │   │     h' = GELU(h') → Dropout                                     │ │    │
│  │   │     h = h + h'                    # Residual connection        │ │    │
│  │   │                                                                 │ │    │
│  │   │   Dimensions: 576 → 512 → 512 → 256 → 256                       │ │    │
│  │   │                                                                 │ │    │
│  │   └─────────────────────────────────────────────────────────────────┘ │    │
│  │                                                                        │    │
│  │   ┌─────────────────────────────────────────────────────────────────┐ │    │
│  │   │                    Output Heads                                 │ │    │
│  │   │                                                                 │ │    │
│  │   │   Main Task:                                                    │ │    │
│  │   │     Δx_pred = Linear(256, G)      # Perturbation response      │ │    │
│  │   │                                                                 │ │    │
│  │   │   Auxiliary Task (C1):                                          │ │    │
│  │   │     z_tme_post = Linear(256, 32)  # Post-perturbation TME      │ │    │
│  │   │                                                                 │ │    │
│  │   └─────────────────────────────────────────────────────────────────┘ │    │
│  │                                                                        │    │
│  └───────────────────────────────────────────────────────────────────────┘    │
│          │                                                                     │
│          ▼                                                                     │
│  ┌───────────────────────────────────────────────────────────────────────┐    │
│  │                          OUTPUT                                        │    │
│  ├───────────────────────────────────────────────────────────────────────┤    │
│  │                                                                        │    │
│  │   Primary Output:                                                      │    │
│  │     Δx_pred ∈ R^{batch×G}        Predicted expression change          │    │
│  │                                                                        │    │
│  │   Auxiliary Outputs:                                                   │    │
│  │     z_cell ∈ R^{batch×512}       Cell embedding                       │    │
│  │     z_tme ∈ R^{batch×32}         TME embedding                        │    │
│  │     h_low ∈ R^{batch×16}         Low-frequency features (A1)          │    │
│  │     h_high ∈ R^{batch×16}        High-frequency features (A1)         │    │
│  │     z_tme_post ∈ R^{batch×32}    Post-perturbation TME (C1)           │    │
│  │                                                                        │    │
│  └───────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Semantic Hyperedge Construction

超边是超图的核心概念。与普通图的边只能连接两个节点不同，超边可以连接任意数量的节点，更适合建模"一组细胞构成一个功能单元"这样的高阶关系。

#### 1.1 Concept: Graph vs Hypergraph

```txt
普通图 (Graph)                      超图 (Hypergraph)
==================                  ==================

    A ─── B                         ┌─────────────────┐
    │     │                         │   A   B   C     │  ← Hyperedge E1
    │     │                         └─────────────────┘
    C ─── D
                                    ┌───────────┐
Edge: 连接2个节点                   │   B   D   │      ← Hyperedge E2
                                    └───────────┘

                                    Hyperedge: 连接任意多个节点
```

#### 1.2 Four Hyperedge Types

SP-HyperRAE 构建四种语义超边，每种编码不同的生物学信息：

```txt
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HYPEREDGE TYPES                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1. T_contact (T细胞接触超边)                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Construction:                                                     │  │
│   │   - Calculate T-cell contact ratio for each cell                   │  │
│   │   - Stratify by ratio bins: [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]  │  │
│   │   - DBSCAN clustering within each bin                              │  │
│   │   - Each cluster → one hyperedge                                   │  │
│   │                                                                     │  │
│   │   Biological meaning:                                               │  │
│   │   - Immune infiltration regions                                     │  │
│   │   - Cells with similar immune microenvironment                     │  │
│   │                                                                     │  │
│   │   Example: Cells at tumor-immune interface with 30-50% T neighbors │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   2. Tumor_contact (肿瘤接触超边)                                           │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Construction:                                                     │  │
│   │   - Calculate tumor cell contact ratio for each cell               │  │
│   │   - Stratify by ratio bins (same as above)                         │  │
│   │   - DBSCAN clustering within each bin                              │  │
│   │   - Each cluster → one hyperedge                                   │  │
│   │                                                                     │  │
│   │   Biological meaning:                                               │  │
│   │   - Tumor invasion fronts                                          │  │
│   │   - Cells experiencing tumor pressure                              │  │
│   │                                                                     │  │
│   │   Example: Stromal cells at tumor boundary with 50-70% tumor neighbors │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   3. Interface (界面超边)                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Construction:                                                     │  │
│   │   - Identify cells with BOTH T-cell AND tumor neighbors            │  │
│   │   - DBSCAN clustering on these interface cells                     │  │
│   │   - Each cluster → one hyperedge                                   │  │
│   │                                                                     │  │
│   │   Biological meaning:                                               │  │
│   │   - Active tumor-immune interaction zones                          │  │
│   │   - Hotspots of immunological activity                             │  │
│   │                                                                     │  │
│   │   Example: Cells at the exact boundary between tumor nest and TILs │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   4. Spatial (空间邻近超边)                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Construction:                                                     │  │
│   │   - For each cell, get its K nearest neighbors                     │  │
│   │   - Cell + neighbors → one hyperedge                               │  │
│   │                                                                     │  │
│   │   Biological meaning:                                               │  │
│   │   - General spatial context                                        │  │
│   │   - Local microenvironment regardless of cell type                 │  │
│   │                                                                     │  │
│   │   Example: Each cell with its 15 nearest neighbors                 │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 1.3 Construction Algorithm

```python
# Pseudocode for hyperedge construction

def build_semantic_hyperedges(coords, cell_types, adj_list, config):
    """
    Build four types of semantic hyperedges

    Args:
        coords: [N, 2] spatial coordinates
        cell_types: [N] cell type labels
        adj_list: adjacency list from KNN graph
        config: configuration object

    Returns:
        hyperedge_dict: {type: [set(nodes), ...]}
        node_to_hyperedges: {node: [(type, idx), ...]}
    """

    # Step 1: Infer cell type mapping
    # T_cell → 0, Tumor → 1, Overlap → 2, Background → 3
    cell_type_map = infer_cell_type_mapping(cell_types)

    # Step 2: Calculate neighborhood features
    for each cell i:
        neighbors = adj_list[i]
        t_ratio[i] = fraction of neighbors that are T cells
        tumor_ratio[i] = fraction of neighbors that are tumor cells
        is_interface[i] = has both T and tumor neighbors

    # Step 3: Build T_contact hyperedges
    for each ratio_bin in [0.01-0.05, 0.05-0.1, 0.1-0.2, ...]:
        cells_in_bin = cells with t_ratio in this bin
        clusters = DBSCAN(cells_in_bin, eps=auto, min_samples=5)
        for each cluster:
            add cluster as a hyperedge

    # Step 4: Build Tumor_contact hyperedges (same as above)

    # Step 5: Build Interface hyperedges
    interface_cells = cells where is_interface == True
    clusters = DBSCAN(interface_cells)
    for each cluster:
        add cluster as a hyperedge

    # Step 6: Build Spatial hyperedges
    for each cell i:
        hyperedge = {i} ∪ set(adj_list[i])
        add hyperedge

    return hyperedge_dict, node_to_hyperedges
```

#### 1.4 Data Structures

```python
# hyperedge_dict structure
hyperedge_dict = {
    't_contact': [
        {0, 5, 12, 23, 45},      # Hyperedge 0: cells in T-rich region 1
        {102, 103, 108, 115},    # Hyperedge 1: cells in T-rich region 2
        ...
    ],
    'tumor_contact': [
        {200, 201, 205, 210},    # Cells in tumor-adjacent region 1
        ...
    ],
    'interface': [
        {50, 51, 52, 60, 61},    # Cells at interface 1
        ...
    ],
    'spatial': [
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},  # Cell 0 + neighbors
        {1, 0, 2, 3, 5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50},  # Cell 1 + neighbors
        ...
    ]
}

# node_to_hyperedges structure
node_to_hyperedges = {
    0: [('t_contact', 0), ('spatial', 0), ('spatial', 1)],
    1: [('spatial', 1), ('spatial', 2)],
    5: [('t_contact', 0), ('spatial', 0), ('spatial', 5)],
    ...
}
```

### 2. Spectral Hypergraph Convolution

#### 2.1 Hypergraph Laplacian

超图拉普拉斯矩阵是谱超图卷积的核心。

**关联矩阵 (Incidence Matrix)**:

$$H \in \mathbb{R}^{N \times M}$$

其中 $H_{ij} = 1$ 如果节点 $i$ 属于超边 $j$，否则 $H_{ij} = 0$。

**度矩阵**:

- 节点度矩阵: $D_v = \text{diag}(H \cdot \mathbf{1}_M)$
- 超边度矩阵: $D_e = \text{diag}(H^T \cdot \mathbf{1}_N)$

**归一化超图拉普拉斯**:

$$L = I - D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}$$

其中 $W$ 是超边权重矩阵（默认为单位矩阵）。

**直观理解**:

```txt
普通图拉普拉斯:      L = I - D^{-1/2} A D^{-1/2}
                          ↑
                     邻接矩阵 A

超图拉普拉斯:        L = I - D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
                          ↑           ↑         ↑
                     关联矩阵 H    超边权重   关联矩阵转置

超图的 "邻接矩阵" 是通过 H H^T 隐式构建的
(H H^T)_{ij} = 节点 i 和 j 共同出现的超边数量
```

#### 2.2 Chebyshev Polynomial Approximation

直接计算谱卷积需要对拉普拉斯矩阵进行特征分解，复杂度为 $O(N^3)$。切比雪夫多项式近似可以避免这一问题。

**谱卷积定义**:

$$h_{out} = g_\theta(L) \cdot h = U g_\theta(\Lambda) U^T \cdot h$$

其中 $L = U \Lambda U^T$ 是特征分解。

**切比雪夫近似**:

$$g_\theta(L) \approx \sum_{k=0}^{K} \theta_k T_k(\tilde{L})$$

其中:

- $\tilde{L} = \frac{2L}{\lambda_{max}} - I$ 是缩放后的拉普拉斯（特征值范围 [-1, 1]）
- $T_k(x)$ 是切比雪夫多项式

**切比雪夫递推**:

$$T_0(x) = 1$$
$$T_1(x) = x$$
$$T_k(x) = 2x \cdot T_{k-1}(x) - T_{k-2}(x)$$

**计算过程**:

```python
def chebyshev_convolution(h, L_tilde, K, theta):
    """
    Chebyshev polynomial approximation of spectral convolution

    Args:
        h: [N, d_in] input features
        L_tilde: [N, N] scaled Laplacian (sparse)
        K: polynomial order
        theta: [K+1, d_in, d_out] learnable coefficients

    Returns:
        h_out: [N, d_out] output features
    """
    # T_0 = I, so T_0 @ h = h
    T_list = [h]

    # T_1 = L_tilde, so T_1 @ h = L_tilde @ h
    if K > 0:
        T_list.append(sparse_mm(L_tilde, h))

    # T_k = 2 * L_tilde * T_{k-1} - T_{k-2}
    for k in range(2, K + 1):
        T_k = 2 * sparse_mm(L_tilde, T_list[-1]) - T_list[-2]
        T_list.append(T_k)

    # Weighted sum: h_out = sum_k theta_k @ T_k @ h
    h_out = sum(T_list[k] @ theta[k] for k in range(K + 1))

    return h_out
```

**复杂度分析**:

| 方法 | 复杂度 |
|------|--------|
| 直接特征分解 | $O(N^3)$ |
| 切比雪夫近似 | $O(K \cdot \|E\| \cdot d)$ |

其中 $|E|$ 是拉普拉斯矩阵的非零元素数量，$d$ 是特征维度。

#### 2.3 Frequency Decomposition

SP-HyperRAE 的一个关键设计是将超图信号分解为低频和高频成分。

**直观理解**:

```txt
超图拉普拉斯的特征值 λ:
- λ 接近 0: 低频，对应图上变化缓慢的信号（全局模式）
- λ 接近 λ_max: 高频，对应图上变化剧烈的信号（局部变化）

类比图像处理:
- 低频 = 图像的整体色调、大块区域
- 高频 = 边缘、纹理、细节
```

**低通/高通滤波器**:

在理想情况下:

- 低通滤波器: $g_{low}(\lambda) = e^{-\lambda / \sigma_{low}^2}$
- 高通滤波器: $g_{high}(\lambda) = 1 - e^{-\lambda / \sigma_{high}^2}$

在SP-HyperRAE中，我们使用两个独立的切比雪夫卷积来学习这些滤波器:

```python
class SpectralHypergraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, config):
        # Low-pass filter (learnable)
        self.low_pass = ChebyshevConv(in_dim, config.freq_dim, K=config.chebyshev_k)

        # High-pass filter (learnable)
        self.high_pass = ChebyshevConv(in_dim, config.freq_dim, K=config.chebyshev_k)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(config.freq_dim * 2 + in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )

    def forward(self, h, L_tilde):
        h_low = self.low_pass(h, L_tilde)   # [N, freq_dim]
        h_high = self.high_pass(h, L_tilde)  # [N, freq_dim]

        h_fused = self.fusion(torch.cat([h_low, h_high, h], dim=-1))

        return h_fused, h_low, h_high
```

**频率正交损失**:

为了确保低频和高频特征确实捕获不同的信息，我们引入频率正交损失:

$$L_{freq} = \frac{1}{B} \sum_{i=1}^{B} \cos^2(h_{low}^{(i)}, h_{high}^{(i)})$$

其中 $\cos(\cdot, \cdot)$ 是余弦相似度。这个损失鼓励低频和高频特征正交。

### 3. Hyperedge Contrastive Learning

超边对比学习通过三个结构约束损失来增强超图表示的质量。

#### 3.1 B1: Intra-hyperedge Cohesion Loss

**目标**: 同一超边内的节点应该有相似的表示。

**直觉**: 属于同一微环境的细胞应该有相似的状态。

**公式**:

$$L_{intra} = \frac{1}{|E|} \sum_{j=1}^{|E|} \frac{1}{|E_j|} \sum_{i \in E_j} \|z_i - \bar{z}_j\|^2$$

其中:

- $E_j$ 是第 $j$ 个超边
- $\bar{z}_j = \frac{1}{|E_j|} \sum_{i \in E_j} z_i$ 是超边的中心表示

**可视化**:

```txt
超边 E_j = {节点1, 节点2, 节点3, 节点4}

在表示空间中:

    z1  •
         \
          •  z̄_j (超边中心)
         /\
    z2  •  • z3
            \
             • z4

L_intra 最小化每个节点到中心的距离
→ 使同一超边内的节点聚集在一起
```

#### 3.2 B2: Type-level Contrastive Loss (InfoNCE)

**目标**: 同类型的超边应该有相似的表示，不同类型的超边应该有不同的表示。

**直觉**: T_contact 超边应该彼此相似，但与 Tumor_contact 超边不同。

**公式**:

$$L_{type} = -\frac{1}{|E|} \sum_{j=1}^{|E|} \log \frac{\sum_{k \in P(j)} \exp(sim(z_j, z_k) / \tau)}{\sum_{k \neq j} \exp(sim(z_j, z_k) / \tau)}$$

其中:

- $z_j$ 是超边 $j$ 的表示（节点表示的均值）
- $P(j) = \{k : type(k) = type(j), k \neq j\}$ 是同类型的其他超边
- $sim(\cdot, \cdot)$ 是余弦相似度
- $\tau$ 是温度参数

**可视化**:

```txt
超边表示空间:

    T_contact超边          Tumor_contact超边
    ┌─────────────┐        ┌─────────────┐
    │  •  •       │        │      •  •   │
    │    •  •     │        │    •  •     │
    │  •    •     │        │  •    •     │
    └─────────────┘        └─────────────┘
           ↓                      ↓
    拉近同类型                拉近同类型
           ↓                      ↓
    ←────────────────推远────────────────→
```

#### 3.3 B3: Inter-hyperedge Separation Loss

**目标**: 不同超边之间应该保持最小距离，防止表示坍塌。

**直觉**: 每个超边都应该有独特的表示，不能所有超边都映射到同一点。

**公式**:

$$L_{inter} = \frac{1}{|E|(|E|-1)} \sum_{j \neq k} \max(0, m - \|z_j - z_k\|)^2$$

其中 $m$ 是 margin 参数（默认 0.5）。

**可视化**:

```txt
没有 L_inter:                有 L_inter:
所有超边可能坍塌到一点        超边保持最小距离

        •••                     •       •
        •••                       •   •
        •••                     •   •   •
                                  •   •
```

#### 3.4 Combined Contrastive Learning

```python
class HyperedgeContrastiveLoss(nn.Module):
    def forward(self, z_nodes, hyperedge_dict):
        """
        Args:
            z_nodes: [N, dim] node representations
            hyperedge_dict: {type: [set(nodes), ...]}

        Returns:
            losses: dict of loss values
        """
        # Compute hyperedge embeddings (mean pooling)
        edge_embeddings = {}
        for etype, edges in hyperedge_dict.items():
            embs = []
            for edge_nodes in edges:
                emb = z_nodes[list(edge_nodes)].mean(dim=0)
                embs.append(emb)
            edge_embeddings[etype] = torch.stack(embs)

        # B1: Intra-hyperedge cohesion
        L_intra = self.intra_hyperedge_loss(z_nodes, hyperedge_dict)

        # B2: Type-level contrast
        L_type = self.type_contrastive_loss(edge_embeddings)

        # B3: Inter-hyperedge separation
        L_inter = self.inter_hyperedge_loss(edge_embeddings)

        return {
            'intra': L_intra,
            'type': L_type,
            'inter': L_inter,
            'total': α1*L_intra + α2*L_type + α3*L_inter
        }
```

### 4. Multi-Task Decoder

#### 4.1 FiLM Modulation

FiLM (Feature-wise Linear Modulation) 是一种条件生成技术，允许TME信息影响解码过程。

**公式**:

$$h' = \gamma \odot \text{LayerNorm}(h) + \beta$$

其中 $\gamma, \beta$ 由 TME 嵌入生成:

$$[\gamma; \beta] = \text{Linear}(z_{tme})$$

**直觉**: TME 状态决定了解码器如何处理信息，类似于"在这个微环境下，细胞应该如何响应扰动"。

#### 4.2 Residual Decoder Block

```python
class ResidualDecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, tme_dim, dropout=0.1):
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # FiLM parameters
        self.film = nn.Linear(tme_dim, out_dim * 2)

        # Residual projection if needed
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, h, z_tme):
        # Main path
        h_new = self.linear(h)
        h_new = self.norm(h_new)

        # FiLM modulation
        gamma, beta = self.film(z_tme).chunk(2, dim=-1)
        h_new = gamma * h_new + beta

        h_new = self.act(h_new)
        h_new = self.dropout(h_new)

        # Residual connection
        return self.residual(h) + h_new
```

#### 4.3 Multi-Task Output

```python
class MultiTaskDecoder(nn.Module):
    def forward(self, z_cell, z_tme, perturb_idx, z_perturb_guided):
        # Concatenate all embeddings
        h = torch.cat([z_cell, z_tme, z_perturb_guided], dim=-1)  # [batch, 576]

        # Pass through residual blocks
        for block in self.decoder_blocks:
            h = block(h, z_tme)

        # Output heads
        delta_x_pred = self.main_head(h)      # [batch, n_genes]
        z_tme_post = self.tme_head(h)         # [batch, 32]

        return {
            'delta_x_pred': delta_x_pred,
            'z_tme_post': z_tme_post
        }
```

### 5. Loss Function

#### 5.1 Complete Loss Function

$$L_{total} = \underbrace{L_{recon}}_{\text{主任务}} + \underbrace{\alpha_1 L_{intra} + \alpha_2 L_{type} + \alpha_3 L_{inter}}_{\text{超边对比 (B1+B2+B3)}} + \underbrace{\alpha_4 L_{freq}}_{\text{频率正交 (A1)}} + \underbrace{\alpha_5 L_{TME}}_{\text{TME多样性}} + \underbrace{\alpha_6 L_{KL}}_{\text{VAE正则}}$$

#### 5.2 Loss Components

| Loss | Formula | Weight | Purpose |
|------|---------|--------|---------|
| $L_{recon}$ | $\text{MSE}(\Delta x_{pred}, \Delta x_{true})$ | 1.0 | Main task: predict expression change |
| $L_{intra}$ | $\frac{1}{\|E\|} \sum_j \frac{1}{\|E_j\|} \sum_{i \in E_j} \|z_i - \bar{z}_j\|^2$ | 0.1 | B1: intra-hyperedge cohesion |
| $L_{type}$ | InfoNCE across hyperedge types | 0.05 | B2: type-level contrast |
| $L_{inter}$ | $\frac{1}{\|E\|(\|E\|-1)} \sum_{j \neq k} \max(0, m - \|z_j - z_k\|)^2$ | 0.01 | B3: inter-hyperedge separation |
| $L_{freq}$ | $\frac{1}{B} \sum_i \cos^2(h_{low}^{(i)}, h_{high}^{(i)})$ | 0.05 | A1: frequency orthogonality |
| $L_{TME}$ | Off-diagonal similarity minimization | 0.1 | TME diversity |
| $L_{KL}$ | $-\frac{1}{2}(1 + \log\sigma^2 - \mu^2 - \sigma^2)$ | 0.0001 | VAE regularization |

#### 5.3 Loss Implementation

```python
class SPHyperRAELoss(nn.Module):
    def forward(self, pred, target, mu, logvar, z_tme, h_low, h_high, z_nodes, hyperedge_dict):
        losses = {}

        # Main task loss
        losses['recon'] = F.mse_loss(pred, target)

        # KL divergence (VAE)
        if mu is not None and logvar is not None:
            losses['kl'] = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # TME diversity
        if z_tme is not None:
            z_norm = F.normalize(z_tme, dim=-1)
            sim = torch.mm(z_norm, z_norm.t())
            mask = torch.eye(sim.size(0), device=sim.device).bool()
            losses['tme_diversity'] = sim[~mask].pow(2).mean()

        # Frequency orthogonality (A1)
        if h_low is not None and h_high is not None:
            h_low_norm = F.normalize(h_low, dim=-1)
            h_high_norm = F.normalize(h_high, dim=-1)
            cos_sim = (h_low_norm * h_high_norm).sum(dim=-1)
            losses['freq'] = (cos_sim ** 2).mean()

        # Hyperedge contrastive losses (B1+B2+B3)
        if z_nodes is not None and hyperedge_dict is not None:
            contrast_losses = self.hyperedge_contrast_loss(z_nodes, hyperedge_dict)
            losses.update(contrast_losses)

        # Weighted total
        losses['total'] = (
            self.recon_weight * losses['recon'] +
            self.kl_weight * losses.get('kl', 0) +
            self.tme_diversity_weight * losses.get('tme_diversity', 0) +
            self.freq_weight * losses.get('freq', 0) +
            self.intra_weight * losses.get('intra', 0) +
            self.type_weight * losses.get('type', 0) +
            self.inter_weight * losses.get('inter', 0)
        )

        return losses
```

### 6. Training Strategy

#### 6.1 Optimization

```python
# Optimizer configuration
optimizer = torch.optim.Adam([
    {'params': model.cell_encoder.parameters(), 'lr': config.lr * config.scgpt_lr_scale},  # Lower LR for pre-trained
    {'params': model.tme_encoder.parameters(), 'lr': config.lr},
    {'params': model.decoder.parameters(), 'lr': config.lr},
], weight_decay=config.weight_decay)

# Learning rate schedule with warmup
def get_lr(epoch):
    if epoch < config.warmup_epochs:
        return config.lr * (epoch + 1) / config.warmup_epochs
    return config.lr

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
```

#### 6.2 Training Loop

```python
def train_model(model, train_loader, config, hyperedge_dict, all_expression):
    model.train()

    for epoch in range(config.epochs):
        for batch in train_loader:
            # Forward pass
            output = model(
                x=batch['expression'],
                perturb_idx=batch['perturb_idx'],
                global_center_indices=batch['global_indices'],
                all_expression=all_expression
            )

            # Compute losses
            losses = loss_fn(
                pred=output['delta_x_pred'],
                target=batch['delta_x'],
                mu=output['mu'],
                logvar=output['logvar'],
                z_tme=output['z_tme'],
                h_low=output['h_low'],
                h_high=output['h_high'],
                z_nodes=output['z_tme'],  # Use TME embeddings for contrastive
                hyperedge_dict=hyperedge_dict
            )

            # Backward pass
            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
```

#### 6.3 scGPT Fine-tuning Strategy

```txt
Epoch 1-5 (Warmup):
├── scGPT: Frozen (requires_grad=False)
├── TME Encoder: Training with warmup LR
├── Decoder: Training with warmup LR
└── Purpose: Let other modules adapt to frozen scGPT features

Epoch 6+ (Full training):
├── scGPT: Training with 0.1× LR (optional)
├── TME Encoder: Training with full LR
├── Decoder: Training with full LR
└── Purpose: Fine-tune all components together
```

---

## Installation

### System Requirements

- **Operating System**: Linux, macOS, or Windows
- **Python**: >= 3.8
- **RAM**: >= 16GB (recommended 32GB for large datasets)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

### Step 1: Create Environment

```bash
# Using conda (recommended)
conda create -n sp_hyperrae python=3.9
conda activate sp_hyperrae

# Or using venv
python -m venv sp_hyperrae_env
source sp_hyperrae_env/bin/activate  # Linux/macOS
# sp_hyperrae_env\Scripts\activate   # Windows
```

### Step 2: Install PyTorch

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision torchaudio

# macOS with MPS
pip install torch torchvision torchaudio
```

### Step 3: Install Dependencies

```bash
# Core dependencies
pip install numpy scipy scikit-learn pandas

# Single-cell analysis
pip install anndata scanpy

# Visualization
pip install matplotlib seaborn

# Progress bars
pip install tqdm

# Optional: Jupyter support
pip install jupyter ipykernel
python -m ipykernel install --user --name sp_hyperrae
```

### Step 4: Install scGPT (Optional)

```bash
# Install scGPT package
pip install scgpt

# Download pre-trained model
mkdir -p scGPT/save/scGPT_human
cd scGPT/save/scGPT_human

# Download from official source (adjust URL as needed)
# Required files:
# - vocab.json
# - args.json
# - best_model.pt
```

### Step 5: Verify Installation

```python
# Test imports
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

import sp_hyperrae
print(f"SP-HyperRAE version: {sp_hyperrae.__version__}")
```

---

## Quick Start

### Command Line Interface

```bash
# Basic run
python -m sp_hyperrae.main --data your_data.h5ad

# With spectral hypergraph convolution (A1)
python -m sp_hyperrae.main --data your_data.h5ad --use_spectral

# With hyperedge contrastive learning (B1+B2+B3)
python -m sp_hyperrae.main --data your_data.h5ad --use_contrast

# Full features
python -m sp_hyperrae.main --data your_data.h5ad \
    --use_spectral \
    --use_contrast \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4

# Simple mode (no hypergraph, just MLP)
python -m sp_hyperrae.main --data your_data.h5ad --simple
```

### Python API - Basic Example

```python
import torch
from sp_hyperrae import SPHyperRAE, ModelConfig
from sp_hyperrae.data import load_and_preprocess_data, create_dataloaders
from sp_hyperrae.hypergraph import build_spatial_graph, build_semantic_hyperedges
from sp_hyperrae.train import train_model, evaluate_model

# 1. Configuration
config = ModelConfig(
    n_genes=500,
    epochs=50,
    batch_size=32,
    use_spectral_conv=True,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# 2. Load data
data_dict = load_and_preprocess_data(config)

# 3. Build spatial graph
edge_index, edge_attr, adj_list = build_spatial_graph(
    data_dict['coords_all'],
    k=config.k_neighbors,
    sample_labels=data_dict.get('sample_labels_all')
)

# 4. Build semantic hyperedges
hyperedge_dict, node_to_hyperedges, _ = build_semantic_hyperedges(
    data_dict['coords_all'],
    data_dict['cell_types_all'],
    adj_list,
    config
)

# 5. Create data loaders
train_loader, test_loader = create_dataloaders(
    data_dict, edge_index, edge_attr, config,
    hyperedge_dict=hyperedge_dict,
    node_to_hyperedges=node_to_hyperedges
)

# 6. Initialize model
model = SPHyperRAE(
    n_genes=config.n_genes,
    n_perturbations=len(data_dict['perturb_names']),
    config=config
).to(config.device)

# 7. Set hypergraph structure
model.set_hypergraph(
    hyperedge_dict,
    node_to_hyperedges,
    n_nodes=data_dict['n_cells_all'],
    device=config.device
)

# 8. Train
model = train_model(
    model, train_loader, config,
    hyperedge_dict=hyperedge_dict,
    all_expression=data_dict['expression_all'],
    use_full_loss=True
)

# 9. Evaluate
results = evaluate_model(model, test_loader, data_dict, config)
print(f"Test MSE: {results['mse']:.4f}")
print(f"Test Pearson (gene): {results['pearson_gene']:.4f}")
print(f"Test Pearson (cell): {results['pearson_cell']:.4f}")
```

### Python API - Advanced Example

```python
import torch
import numpy as np
from sp_hyperrae import (
    SPHyperRAE, ModelConfig,
    HypergraphLaplacian, SpectralHypergraphConv,
    HyperedgeContrastiveLoss
)

# Custom configuration
config = ModelConfig(
    # Data
    n_genes=1000,
    k_neighbors=20,

    # Model architecture
    tme_hidden_dim=64,
    tme_embed_dim=64,
    n_hypergraph_layers=4,
    chebyshev_k=5,

    # Training
    epochs=100,
    batch_size=64,
    lr=5e-5,
    warmup_epochs=10,

    # Loss weights
    recon_weight=1.0,
    intra_weight=0.2,
    type_weight=0.1,
    inter_weight=0.02,
    freq_weight=0.1,

    # Device
    device='cuda'
)

# Custom hyperedge construction
def custom_hyperedge_builder(coords, cell_types, adj_list):
    """Build custom hyperedges based on your data"""
    hyperedge_dict = {
        'custom_type_1': [],
        'custom_type_2': [],
    }

    # Your custom logic here
    # ...

    return hyperedge_dict, node_to_hyperedges

# Build custom hypergraph
hyperedge_dict, node_to_hyperedges = custom_hyperedge_builder(
    coords, cell_types, adj_list
)

# Custom training loop with logging
def custom_train(model, train_loader, config, hyperedge_dict, all_expression):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    loss_history = {'total': [], 'recon': [], 'intra': [], 'type': [], 'inter': [], 'freq': []}

    for epoch in range(config.epochs):
        model.train()
        epoch_losses = {k: [] for k in loss_history}

        for batch in train_loader:
            # Move to device
            batch = {k: v.to(config.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward
            output = model(
                x=batch['expression'],
                perturb_idx=batch['perturb_idx'],
                global_center_indices=batch['global_indices'],
                all_expression=torch.tensor(all_expression).to(config.device)
            )

            # Compute losses
            losses = compute_all_losses(output, batch, hyperedge_dict, config)

            # Backward
            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Log
            for k, v in losses.items():
                epoch_losses[k].append(v.item())

        # Epoch summary
        for k in loss_history:
            loss_history[k].append(np.mean(epoch_losses[k]))

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={loss_history['total'][-1]:.4f}")

    return model, loss_history

# Train with custom loop
model, loss_history = custom_train(model, train_loader, config, hyperedge_dict, all_expression)

# Extract embeddings for downstream analysis
model.eval()
with torch.no_grad():
    all_z_tme = model.get_tme_embeddings(torch.tensor(all_expression).to(config.device))
    all_z_tme = all_z_tme.cpu().numpy()

# Visualize TME embeddings
import scanpy as sc
import matplotlib.pyplot as plt

adata = sc.AnnData(all_z_tme)
sc.pp.neighbors(adata, n_neighbors=15)
sc.tl.umap(adata)
sc.pl.umap(adata, color='leiden')
```

---

## Data Format

### Input: AnnData (.h5ad)

SP-HyperRAE expects input data in AnnData format with the following structure:

#### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `adata.X` | ndarray/sparse | Gene expression matrix (cells × genes) |
| `adata.obs['cell_type']` | categorical | Cell type annotations |
| `adata.obs['perturbation']` | categorical | Perturbation labels |
| `adata.obsm['spatial']` | ndarray | Spatial coordinates (N × 2) |

#### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `adata.obs['sample']` | categorical | Sample/batch labels for multi-sample data |
| `adata.obs['delta_expression']` | ndarray | Pre-computed expression changes (if available) |
| `adata.var['highly_variable']` | bool | Pre-selected highly variable genes |

### Example: Creating Input Data

```python
import anndata as ad
import numpy as np
import pandas as pd

# Simulate data
n_cells = 5000
n_genes = 2000

# Expression matrix
X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes)).astype(np.float32)

# Cell metadata
obs = pd.DataFrame({
    'cell_type': np.random.choice(['T_cell', 'Tumor', 'Stromal', 'Macrophage'], n_cells),
    'perturbation': np.random.choice(['Control', 'DrugA', 'DrugB', 'DrugC'], n_cells),
    'sample': np.random.choice(['Patient1', 'Patient2', 'Patient3'], n_cells),
})

# Spatial coordinates (simulate tissue structure)
spatial = np.zeros((n_cells, 2))
for i, ct in enumerate(obs['cell_type']):
    if ct == 'Tumor':
        spatial[i] = np.random.randn(2) * 50 + [0, 0]  # Tumor core
    elif ct == 'T_cell':
        spatial[i] = np.random.randn(2) * 30 + [100, 0]  # Immune zone
    else:
        spatial[i] = np.random.randn(2) * 80  # Dispersed

# Gene names
var = pd.DataFrame(index=[f'Gene_{i}' for i in range(n_genes)])

# Create AnnData
adata = ad.AnnData(X=X, obs=obs, var=var)
adata.obsm['spatial'] = spatial

# Save
adata.write('example_spatial_perturbation.h5ad')
print(f"Saved AnnData with shape {adata.shape}")
```

### Data Preprocessing Pipeline

```python
from sp_hyperrae.data import load_and_preprocess_data

# The preprocessing pipeline:
# 1. Load h5ad file
# 2. Filter cells with perturbation labels
# 3. Select highly variable genes (or use provided)
# 4. Normalize: log1p(X / lib_size * 10000)
# 5. Extract spatial coordinates
# 6. Map cell types
# 7. Split train/test

data_dict = load_and_preprocess_data(config)

# data_dict contains:
# - 'expression_all': [N, G] normalized expression
# - 'expression': [N_filtered, G] filtered expression (with perturbation labels)
# - 'coords_all': [N, 2] spatial coordinates
# - 'cell_types_all': [N] cell type labels
# - 'perturb_labels': [N_filtered] perturbation indices
# - 'perturb_names': list of perturbation names
# - 'gene_names': list of gene names
# - 'sample_labels_all': [N] sample labels (if available)
# - 'train_idx', 'test_idx': train/test indices
```

---

## Project Structure

```txt
sp_hyperrae/
│
├── __init__.py                 # Package initialization and exports
│   - Defines __version__, __all__
│   - Imports all public classes and functions
│
├── config.py                   # Configuration management
│   - Config: base configuration
│   - LossConfig: loss function weights
│   - SpectralConfig: spectral convolution parameters
│   - ModelConfig: combined configuration class
│
├── model.py                    # Main model architecture
│   - SPHyperRAE: complete model class
│   - Forward pass logic
│   - Hypergraph setup
│   - Parameter counting utilities
│
├── spectral_conv.py            # Spectral hypergraph convolution (A1)
│   - HypergraphLaplacian: Laplacian matrix construction
│   - ChebyshevConv: Chebyshev polynomial convolution
│   - SpectralHypergraphConv: frequency decomposition layer
│   - MultiLayerSpectralHypergraphEncoder: multi-layer encoder
│   - compute_frequency_orthogonality_loss: A1 loss function
│
├── hypergraph.py               # Hyperedge construction
│   - build_spatial_graph: KNN graph construction
│   - build_semantic_hyperedges: semantic hyperedge construction
│   - infer_cell_type_mapping: automatic cell type inference
│   - LocalGraphExtractor: K-hop subgraph extraction
│   - get_hyperedge_statistics: hyperedge statistics
│
├── encoders.py                 # Encoder modules
│   - ScGPTCellEncoder: pre-trained scGPT encoder
│   - FallbackCellEncoder: MLP fallback encoder
│   - ScGPTGuidedFusion: cross-attention fusion
│   - TMEPredictor: TME change prediction
│
├── decoder.py                  # Decoder modules
│   - ResidualDecoderBlock: residual block with FiLM
│   - RAEDecoder: main decoder
│   - MultiTaskDecoder: multi-task output heads
│
├── losses.py                   # Loss functions
│   - ReconstructionLoss: MSE loss
│   - KLDivergenceLoss: VAE KL divergence
│   - TMEDiversityLoss: TME diversity regularization
│   - SPHyperRAELoss: complete loss function
│   - SimpleLoss: simplified loss (no hypergraph)
│
├── contrastive.py              # Hyperedge contrastive learning (B1+B2+B3)
│   - HyperedgeContrastiveLoss: combined contrastive loss
│   - intra_hyperedge_loss: B1 loss
│   - type_contrastive_loss: B2 loss (InfoNCE)
│   - inter_hyperedge_loss: B3 loss (margin)
│   - HyperedgeEmbeddingExtractor: hyperedge embedding extraction
│
├── data.py                     # Data loading and processing
│   - load_and_preprocess_data: main data loading function
│   - create_dataloaders: DataLoader creation
│   - SpatialPerturbDataset: custom Dataset class
│   - Gene selection and normalization utilities
│
├── train.py                    # Training and evaluation
│   - train_model: main training loop
│   - evaluate_model: evaluation function
│   - print_results: result formatting
│   - save_results: model and result saving
│
├── main.py                     # Command line interface
│   - Argument parsing
│   - scGPT loading
│   - Main execution flow
│
└── README.md                   # This documentation
```

---

## Configuration

### Complete Configuration Reference

```python
class ModelConfig:
    """Complete configuration with all parameters"""

    # ==================== PATH CONFIGURATION ====================
    data_dir = '.'                          # Data directory
    output_dir = './sp_hyperrae_results'    # Output directory
    data_file = 'data.h5ad'                 # Input data file
    scgpt_model_dir = './scGPT/save/scGPT_human'  # scGPT model path

    # ==================== DATA PARAMETERS ====================
    n_genes = 500           # Number of highly variable genes to use
    k_neighbors = 15        # K for spatial KNN graph
    k_hop = 2              # Hops for local subgraph extraction
    max_seq_len = 1200     # Maximum sequence length for scGPT

    # ==================== HYPEREDGE PARAMETERS ====================
    hyperedge_eps = 2000        # DBSCAN clustering radius
    hyperedge_min_samples = 5   # DBSCAN minimum samples per cluster
    n_hypergraph_layers = 3     # Number of spectral convolution layers
    use_true_hypergraph = True  # Use hypergraph (vs simple graph)
    n_hyperedge_types = 4       # Number of hyperedge types

    # ==================== MODEL ARCHITECTURE ====================
    scgpt_embed_dim = 512       # scGPT embedding dimension
    tme_hidden_dim = 32         # TME encoder hidden dimension
    tme_embed_dim = 32          # TME embedding dimension
    perturb_embed_dim = 32      # Perturbation embedding dimension
    decoder_hidden_dim = 512    # Decoder hidden dimension
    latent_dim = 512            # RAE latent space dimension
    n_decoder_layers = 4        # Number of decoder layers
    n_gnn_layers = 3            # Number of GNN layers (if using GNN)
    dropout = 0.1               # Dropout rate

    # ==================== SPECTRAL CONVOLUTION (A1) ====================
    use_spectral_conv = True    # Enable spectral hypergraph convolution
    chebyshev_k = 3            # Chebyshev polynomial order (K)
    sigma_low = 0.5            # Low-pass filter parameter
    sigma_high = 1.5           # High-pass filter parameter
    freq_dim = 16              # Frequency feature dimension

    # ==================== TRAINING PARAMETERS ====================
    batch_size = 32            # Training batch size
    epochs = 50                # Number of training epochs
    lr = 1e-4                  # Learning rate
    weight_decay = 0.01        # Weight decay (L2 regularization)
    warmup_epochs = 5          # Learning rate warmup epochs
    freeze_scgpt = True        # Freeze scGPT parameters
    scgpt_finetune_layers = 0  # Number of scGPT layers to fine-tune
    scgpt_lr_scale = 0.1       # Learning rate scale for scGPT
    seed = 42                  # Random seed

    # ==================== LOSS WEIGHTS ====================
    recon_weight = 1.0         # Reconstruction loss (main task)
    intra_weight = 0.1         # B1: intra-hyperedge cohesion
    type_weight = 0.05         # B2: type-level contrast
    inter_weight = 0.01        # B3: inter-hyperedge separation
    inter_margin = 0.5         # B3: margin parameter
    tme_diversity_weight = 0.1 # TME diversity loss
    freq_weight = 0.05         # A1: frequency orthogonality
    kl_weight = 0.0001         # VAE KL divergence
    contrast_temperature = 0.1 # B2: InfoNCE temperature

    # ==================== OUTPUT CONTROL ====================
    output_tme_features = True  # Output TME features for analysis
    output_freq_features = True # Output frequency features

    # ==================== DEVICE ====================
    device = 'cuda'  # 'cuda', 'mps', or 'cpu'
```

### Configuration Presets

```python
# Preset 1: Quick experiment (fast, less accurate)
config_quick = ModelConfig(
    n_genes=200,
    epochs=20,
    batch_size=64,
    n_hypergraph_layers=2,
    chebyshev_k=2,
    use_spectral_conv=False
)

# Preset 2: Standard (balanced speed and accuracy)
config_standard = ModelConfig(
    n_genes=500,
    epochs=50,
    batch_size=32,
    n_hypergraph_layers=3,
    chebyshev_k=3,
    use_spectral_conv=True
)

# Preset 3: High quality (slower, more accurate)
config_quality = ModelConfig(
    n_genes=1000,
    epochs=100,
    batch_size=16,
    n_hypergraph_layers=4,
    chebyshev_k=5,
    use_spectral_conv=True,
    lr=5e-5
)

# Preset 4: Large dataset (memory optimized)
config_large = ModelConfig(
    n_genes=500,
    epochs=30,
    batch_size=16,
    k_neighbors=10,
    hyperedge_min_samples=10,
    n_hypergraph_layers=2
)
```

---

## API Reference

### Main Classes

#### SPHyperRAE

```python
class SPHyperRAE(nn.Module):
    """
    Main model class for spatial perturbation prediction.

    Args:
        n_genes (int): Number of genes in the expression matrix
        n_perturbations (int): Number of perturbation types
        config (ModelConfig): Configuration object
        vocab (GeneVocab, optional): scGPT vocabulary
        scgpt_model (TransformerModel, optional): Pre-trained scGPT model

    Methods:
        set_hypergraph(hyperedge_dict, node_to_hyperedges, n_nodes, device):
            Set the hypergraph structure for spectral convolution.

        forward(x, perturb_idx, graph, gene_ids, values, padding_mask,
                global_center_indices, all_expression):
            Forward pass through the model.

        get_tme_embeddings(all_expression):
            Get TME embeddings for all cells.

    Returns (forward):
        dict with keys:
            - 'delta_x_pred': [batch, n_genes] predicted expression change
            - 'z_cell': [batch, 512] cell embedding
            - 'z_tme': [batch, 32] TME embedding
            - 'h_low': [batch, 16] low-frequency features
            - 'h_high': [batch, 16] high-frequency features
            - 'z_tme_post': [batch, 32] post-perturbation TME embedding
            - 'mu', 'logvar': VAE parameters (if enabled)
    """
```

#### HypergraphLaplacian

```python
class HypergraphLaplacian:
    """
    Hypergraph Laplacian matrix construction.

    Args:
        n_nodes (int): Number of nodes
        hyperedge_dict (dict): {type: [set(nodes), ...]}
        node_to_hyperedges (dict): {node: [(type, idx), ...]}

    Attributes:
        H (sparse matrix): Incidence matrix [N, M]
        L (sparse matrix): Laplacian matrix [N, N]
        lambda_max (float): Estimated maximum eigenvalue

    Methods:
        get_scaled_laplacian():
            Returns scaled Laplacian for Chebyshev convolution.

        to_torch(device):
            Convert to PyTorch sparse tensor.
    """
```

#### HyperedgeContrastiveLoss

```python
class HyperedgeContrastiveLoss(nn.Module):
    """
    Combined hyperedge contrastive learning loss (B1+B2+B3).

    Args:
        config (ModelConfig): Configuration with loss weights

    Methods:
        forward(z_nodes, hyperedge_dict, batch_indices=None):
            Compute all contrastive losses.

    Returns:
        dict with keys:
            - 'intra': B1 intra-hyperedge cohesion loss
            - 'type': B2 type-level contrastive loss
            - 'inter': B3 inter-hyperedge separation loss
            - 'total': weighted sum of all losses
    """
```

### Functions

#### Data Loading

```python
def load_and_preprocess_data(config, vocab=None):
    """
    Load and preprocess spatial transcriptomics data.

    Args:
        config (ModelConfig): Configuration object
        vocab (GeneVocab, optional): scGPT vocabulary for gene mapping

    Returns:
        dict with keys:
            - 'expression_all': [N, G] all cell expression
            - 'expression': [N_filtered, G] filtered expression
            - 'coords_all': [N, 2] spatial coordinates
            - 'cell_types_all': [N] cell types
            - 'perturb_labels': [N_filtered] perturbation indices
            - 'perturb_names': list of perturbation names
            - 'gene_names': list of gene names
            - 'train_idx', 'test_idx': data splits
            - etc.
    """

def create_dataloaders(data_dict, edge_index, edge_attr, config, ...):
    """
    Create PyTorch DataLoaders for training and testing.

    Returns:
        train_loader, test_loader: DataLoader objects
    """
```

#### Hypergraph Construction

```python
def build_spatial_graph(coords, k=15, sample_labels=None):
    """
    Build spatial KNN graph.

    Args:
        coords: [N, 2] spatial coordinates
        k: number of neighbors
        sample_labels: [N] sample labels (optional)

    Returns:
        edge_index: [2, E] edge indices
        edge_attr: [E, 3] edge attributes (distance, dx, dy)
        adj_list: adjacency list
    """

def build_semantic_hyperedges(coords, cell_types, adj_list, config, ...):
    """
    Build semantic hyperedges based on cell types and spatial structure.

    Returns:
        hyperedge_dict: {type: [set(nodes), ...]}
        node_to_hyperedges: {node: [(type, idx), ...]}
        neighborhood_features: dict of feature arrays
    """
```

#### Training

```python
def train_model(model, train_loader, config, use_scgpt_input=False,
                hyperedge_dict=None, node_to_hyperedges=None,
                all_expression=None, use_full_loss=True):
    """
    Train the SP-HyperRAE model.

    Args:
        model: SPHyperRAE model
        train_loader: training DataLoader
        config: configuration
        hyperedge_dict: hyperedge structure
        all_expression: full expression matrix for spectral conv
        use_full_loss: use complete loss (vs simplified)

    Returns:
        trained model
    """

def evaluate_model(model, test_loader, data_dict, config, ...):
    """
    Evaluate the model on test data.

    Returns:
        dict with evaluation metrics:
            - 'mse': mean squared error
            - 'pearson_gene': per-gene Pearson correlation
            - 'pearson_cell': per-cell Pearson correlation
            - 'spearman_gene': per-gene Spearman correlation
            - 'spearman_cell': per-cell Spearman correlation
            - 'predictions': predicted values
            - 'targets': true values
    """
```

---

## Experiments

### Recommended Experiment Protocol

#### 1. Ablation Study

Compare different components:

```bash
# Baseline: MLP only
python -m sp_hyperrae.main --data data.h5ad --simple --epochs 50

# + Spectral convolution (A1)
python -m sp_hyperrae.main --data data.h5ad --use_spectral --epochs 50

# + Contrastive learning (B1+B2+B3)
python -m sp_hyperrae.main --data data.h5ad --use_contrast --epochs 50

# Full model (A1 + B1+B2+B3)
python -m sp_hyperrae.main --data data.h5ad --use_spectral --use_contrast --epochs 50
```

#### 2. Hyperparameter Sensitivity

```python
# Chebyshev order K
for k in [1, 2, 3, 5, 7]:
    config = ModelConfig(chebyshev_k=k)
    # train and evaluate

# Number of spectral layers
for n_layers in [1, 2, 3, 4, 5]:
    config = ModelConfig(n_hypergraph_layers=n_layers)
    # train and evaluate

# Loss weight sensitivity
for intra_w in [0.01, 0.05, 0.1, 0.2, 0.5]:
    config = ModelConfig(intra_weight=intra_w)
    # train and evaluate
```

#### 3. Baseline Comparisons

| Method | Description | Implementation |
|--------|-------------|----------------|
| MLP | Simple feedforward network | `--simple` flag |
| GCN | Graph convolutional network | Replace hypergraph with graph |
| GAT | Graph attention network | Custom implementation |
| HGNN | Standard hypergraph NN | Remove frequency decomposition |
| scGPT | Pre-trained only | Freeze and evaluate |

### Evaluation Metrics

```python
def compute_metrics(predictions, targets):
    """
    Compute all evaluation metrics.
    """
    # MSE
    mse = np.mean((predictions - targets) ** 2)

    # Per-gene Pearson correlation
    pearson_gene = []
    for g in range(predictions.shape[1]):
        r, _ = pearsonr(predictions[:, g], targets[:, g])
        if not np.isnan(r):
            pearson_gene.append(r)
    pearson_gene_mean = np.mean(pearson_gene)

    # Per-cell Pearson correlation
    pearson_cell = []
    for i in range(predictions.shape[0]):
        r, _ = pearsonr(predictions[i], targets[i])
        if not np.isnan(r):
            pearson_cell.append(r)
    pearson_cell_mean = np.mean(pearson_cell)

    # Spearman correlations
    spearman_gene = []
    for g in range(predictions.shape[1]):
        r, _ = spearmanr(predictions[:, g], targets[:, g])
        if not np.isnan(r):
            spearman_gene.append(r)
    spearman_gene_mean = np.mean(spearman_gene)

    return {
        'mse': mse,
        'pearson_gene': pearson_gene_mean,
        'pearson_cell': pearson_cell_mean,
        'spearman_gene': spearman_gene_mean,
    }
```

---

## Visualization

### TME Embedding Visualization

```python
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np

# Get TME embeddings
model.eval()
with torch.no_grad():
    z_tme = model.get_tme_embeddings(
        torch.tensor(data_dict['expression_all']).to(config.device)
    ).cpu().numpy()

# Create AnnData for visualization
adata_vis = sc.AnnData(z_tme)
adata_vis.obs['cell_type'] = data_dict['cell_types_all']
adata_vis.obsm['spatial'] = data_dict['coords_all']

# UMAP
sc.pp.neighbors(adata_vis, n_neighbors=15)
sc.tl.umap(adata_vis)

# Plot UMAP colored by cell type
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sc.pl.umap(adata_vis, color='cell_type', ax=axes[0], show=False, title='TME Embedding (UMAP)')
sc.pl.embedding(adata_vis, basis='spatial', color='cell_type', ax=axes[1], show=False, title='Spatial')
plt.tight_layout()
plt.savefig('tme_embedding.png', dpi=150)
```

### Frequency Features Visualization

```python
# Get frequency features
model.eval()
with torch.no_grad():
    output = model(
        x=torch.tensor(data_dict['expression']).to(config.device),
        perturb_idx=torch.tensor(data_dict['perturb_labels']).to(config.device),
        global_center_indices=torch.arange(len(data_dict['expression'])).to(config.device),
        all_expression=torch.tensor(data_dict['expression_all']).to(config.device)
    )
    h_low = output['h_low'].cpu().numpy()
    h_high = output['h_high'].cpu().numpy()

# Visualize low vs high frequency
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Low frequency - should show global patterns
adata_low = sc.AnnData(h_low)
adata_low.obsm['spatial'] = data_dict['coords'][:len(h_low)]
sc.pp.neighbors(adata_low)
sc.tl.umap(adata_low)
sc.pl.umap(adata_low, color=adata_low.obs_names, ax=axes[0], show=False,
           title='Low-frequency Features (Global Patterns)')

# High frequency - should show local variations
adata_high = sc.AnnData(h_high)
adata_high.obsm['spatial'] = data_dict['coords'][:len(h_high)]
sc.pp.neighbors(adata_high)
sc.tl.umap(adata_high)
sc.pl.umap(adata_high, color=adata_high.obs_names, ax=axes[1], show=False,
           title='High-frequency Features (Local Variations)')

plt.tight_layout()
plt.savefig('frequency_features.png', dpi=150)
```

### Hyperedge Visualization

```python
import networkx as nx

def visualize_hyperedges(coords, hyperedge_dict, cell_types, sample_idx=0, max_edges=50):
    """
    Visualize hyperedges on spatial coordinates.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    colors = {'T_cell': 'blue', 'Tumor': 'red', 'Stromal': 'green', 'Other': 'gray'}

    for idx, (etype, ax) in enumerate(zip(['t_contact', 'tumor_contact', 'interface', 'spatial'],
                                           axes.flatten())):
        # Plot all cells
        for ct in np.unique(cell_types):
            mask = cell_types == ct
            ax.scatter(coords[mask, 0], coords[mask, 1],
                      c=colors.get(ct, 'gray'), alpha=0.3, s=5, label=ct)

        # Highlight hyperedges
        edges = hyperedge_dict.get(etype, [])[:max_edges]
        for edge_nodes in edges:
            edge_coords = coords[list(edge_nodes)]
            # Draw convex hull or centroid
            centroid = edge_coords.mean(axis=0)
            ax.scatter(centroid[0], centroid[1], c='black', s=100, marker='x')

            # Connect to centroid
            for node_coord in edge_coords:
                ax.plot([centroid[0], node_coord[0]], [centroid[1], node_coord[1]],
                       'k-', alpha=0.1)

        ax.set_title(f'{etype} ({len(edges)} hyperedges)')
        ax.set_xlabel('Spatial X')
        ax.set_ylabel('Spatial Y')

    plt.tight_layout()
    plt.savefig('hyperedge_visualization.png', dpi=150)

visualize_hyperedges(data_dict['coords_all'], hyperedge_dict, data_dict['cell_types_all'])
```

### Training Curves

```python
def plot_training_curves(loss_history):
    """
    Plot training loss curves.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    loss_names = ['total', 'recon', 'intra', 'type', 'inter', 'freq']

    for ax, name in zip(axes.flatten(), loss_names):
        if name in loss_history:
            ax.plot(loss_history[name], label=name)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'{name.upper()} Loss')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
```

### Prediction Scatter Plot

```python
def plot_prediction_scatter(predictions, targets, gene_names, top_k=6):
    """
    Scatter plot of predictions vs targets for top genes.
    """
    # Find genes with highest variance
    var_per_gene = np.var(targets, axis=0)
    top_genes = np.argsort(var_per_gene)[-top_k:]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for ax, gene_idx in zip(axes.flatten(), top_genes):
        pred = predictions[:, gene_idx]
        true = targets[:, gene_idx]

        ax.scatter(true, pred, alpha=0.3, s=10)

        # Add diagonal line
        min_val = min(true.min(), pred.min())
        max_val = max(true.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        # Compute correlation
        r, _ = pearsonr(pred, true)

        ax.set_xlabel('True Expression Change')
        ax.set_ylabel('Predicted Expression Change')
        ax.set_title(f'{gene_names[gene_idx]} (r={r:.3f})')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('prediction_scatter.png', dpi=150)
```

---

## Biological Interpretation

### Frequency Decomposition Interpretation

| Feature | Spectral Property | Biological Meaning | Examples |
|---------|-------------------|-------------------|----------|
| **h_low** | Low eigenvalues (λ→0) | Global tissue organization | Tumor core vs immune zone vs stroma; Large-scale tissue architecture |
| **h_high** | High eigenvalues (λ→λ_max) | Local microenvironment changes | Invasion fronts; Infiltration hotspots; Boundary effects |

**Intuition**:

- Low-frequency signals change slowly across the tissue → capture broad regional patterns
- High-frequency signals change rapidly → capture sharp boundaries and local heterogeneity

**Validation approaches**:

1. Visualize h_low and h_high on spatial coordinates
2. Correlate with known biological annotations (tumor grade, immune score)
3. Compare variance within vs between tissue regions

### Hyperedge Type Interpretation

| Type | Construction Basis | Expected Biology |
|------|-------------------|------------------|
| **T_contact** | T-cell neighbor ratio | Immune infiltration zones; Cells experiencing immune pressure |
| **Tumor_contact** | Tumor cell neighbor ratio | Cells at tumor invasion front; Tumor-influenced regions |
| **Interface** | Both T and Tumor neighbors | Active immune-tumor interaction; Immunological hotspots |
| **Spatial** | K-nearest neighbors | General local context; Baseline spatial correlation |

**Key insight**: Cells in the same hyperedge share a similar microenvironment, so they should respond similarly to perturbations.

### TME Embedding Interpretation

The TME embedding (z_tme) captures the local microenvironment context of each cell. Cells with similar z_tme are in similar microenvironments, even if their cell types differ.

**Downstream analyses**:

1. Cluster cells by z_tme to identify microenvironment niches
2. Correlate z_tme with clinical outcomes
3. Compare z_tme before and after perturbation (z_tme_post)

### Model Interpretability

```python
# Example: Analyze which hyperedge types contribute most
def analyze_hyperedge_importance(model, data_dict, hyperedge_dict):
    """
    Analyze the importance of different hyperedge types.
    """
    model.eval()

    # Train with only specific hyperedge types
    results = {}
    for etype in ['t_contact', 'tumor_contact', 'interface', 'spatial']:
        # Create filtered hyperedge dict
        filtered_dict = {etype: hyperedge_dict[etype]}

        # Evaluate
        metrics = evaluate_with_hyperedges(model, data_dict, filtered_dict)
        results[etype] = metrics

    # Compare performance
    print("Performance by hyperedge type:")
    for etype, metrics in results.items():
        print(f"  {etype}: MSE={metrics['mse']:.4f}, Pearson={metrics['pearson']:.4f}")

    return results
```

---

## Extending SP-HyperRAE

### Adding New Hyperedge Types

```python
def build_custom_hyperedges(coords, features, adj_list, config):
    """
    Build custom hyperedges based on your data.
    """
    hyperedge_dict = {
        'custom_type': []
    }
    node_to_hyperedges = {i: [] for i in range(len(coords))}

    # Your custom logic
    # Example: cluster cells by expression similarity
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=50)
    clusters = kmeans.fit_predict(features)

    for c in range(50):
        edge_nodes = set(np.where(clusters == c)[0])
        if len(edge_nodes) >= 5:
            edge_idx = len(hyperedge_dict['custom_type'])
            hyperedge_dict['custom_type'].append(edge_nodes)
            for node in edge_nodes:
                node_to_hyperedges[node].append(('custom_type', edge_idx))

    return hyperedge_dict, node_to_hyperedges
```

### Custom Loss Functions

```python
class CustomLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.base_loss = SPHyperRAELoss(config)

    def forward(self, output, batch, hyperedge_dict):
        # Get base losses
        losses = self.base_loss(
            pred=output['delta_x_pred'],
            target=batch['delta_x'],
            z_tme=output['z_tme'],
            h_low=output['h_low'],
            h_high=output['h_high'],
            z_nodes=output['z_tme'],
            hyperedge_dict=hyperedge_dict
        )

        # Add custom loss
        # Example: consistency loss between z_tme and z_tme_post
        if output['z_tme_post'] is not None:
            consistency_loss = F.mse_loss(output['z_tme'], output['z_tme_post'])
            losses['consistency'] = consistency_loss
            losses['total'] = losses['total'] + 0.1 * consistency_loss

        return losses
```

### Custom Encoder

```python
class CustomCellEncoder(nn.Module):
    """
    Custom cell encoder (e.g., using a different pre-trained model).
    """
    def __init__(self, n_genes, config):
        super().__init__()

        # Your architecture
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
        )

    def forward(self, x):
        return self.encoder(x)

# Use in model
class CustomSPHyperRAE(SPHyperRAE):
    def __init__(self, n_genes, n_perturbations, config):
        super().__init__(n_genes, n_perturbations, config)

        # Replace cell encoder
        self.cell_encoder = CustomCellEncoder(n_genes, config)
        self.use_scgpt = False
```

---

## Data Source

- GSE193460 <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE193460>
- GSE245582 <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE245582>
- GSE
  - <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE221321>
  - <https://download.brainimagelibrary.org/0c/bd/0cbd479c521afff9/extras/tumors/>

## Acknowledgments

We thank the developers of the following projects:

- **[scGPT](https://github.com/bowang-lab/scGPT)**: Pre-trained single-cell language model
- **[PyTorch](https://pytorch.org/)**: Deep learning framework
- **[Scanpy](https://scanpy.readthedocs.io/)**: Single-cell analysis toolkit
- **[AnnData](https://anndata.readthedocs.io/)**: Annotated data matrices
- **[scikit-learn](https://scikit-learn.org/)**: Machine learning utilities

---
