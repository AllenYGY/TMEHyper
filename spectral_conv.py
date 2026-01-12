#!/usr/bin/env python3
"""
谱超图卷积模块 (A1)

核心创新:
1. 超图拉普拉斯矩阵构建
2. 切比雪夫多项式近似（避免显式特征分解）
3. 频率滤波器（低通/高通）
4. 频率特征融合

生物学意义:
- 低频 (h_low): 全局组织分区（肿瘤核心、免疫区、基质区）
- 高频 (h_high): 局部微环境变化（边界、浸润热点）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix, diags, eye


class HypergraphLaplacian:
    """
    超图拉普拉斯矩阵构建器

    L = I - D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}

    其中:
    - H: 关联矩阵 [N, M]
    - D_v: 节点度矩阵
    - D_e: 超边度矩阵
    - W: 超边权重矩阵
    """

    def __init__(self, n_nodes, hyperedge_dict, node_to_hyperedges):
        """
        Args:
            n_nodes: 节点数量
            hyperedge_dict: {type: [set of node indices, ...]}
            node_to_hyperedges: {node_idx: [(type, edge_idx), ...]}
        """
        self.n_nodes = n_nodes
        self.hyperedge_dict = hyperedge_dict
        self.node_to_hyperedges = node_to_hyperedges

        # 构建关联矩阵和拉普拉斯
        self._build_incidence_matrix()
        self._build_laplacian()

    def _build_incidence_matrix(self):
        """构建超图关联矩阵 H"""
        # 收集所有超边
        all_edges = []
        edge_types = []
        for etype, edges in self.hyperedge_dict.items():
            for edge_nodes in edges:
                all_edges.append(edge_nodes)
                edge_types.append(etype)

        self.n_edges = len(all_edges)
        self.edge_types = edge_types

        if self.n_edges == 0:
            # 空超图，创建单位矩阵
            self.H = eye(self.n_nodes, format="csr")
            return

        # 构建稀疏关联矩阵
        rows, cols, data = [], [], []
        for edge_idx, edge_nodes in enumerate(all_edges):
            for node_idx in edge_nodes:
                rows.append(node_idx)
                cols.append(edge_idx)
                data.append(1.0)

        self.H = csr_matrix((data, (rows, cols)), shape=(self.n_nodes, self.n_edges))

    def _build_laplacian(self):
        """构建归一化超图拉普拉斯矩阵"""
        if self.n_edges == 0:
            self.L = eye(self.n_nodes, format="csr")
            self.lambda_max = 2.0
            return

        H = self.H

        # 节点度: D_v[i] = sum_j H[i,j]
        d_v = np.array(H.sum(axis=1)).flatten()
        d_v[d_v == 0] = 1  # 防止除零

        # 超边度: D_e[j] = sum_i H[i,j]
        d_e = np.array(H.sum(axis=0)).flatten()
        d_e[d_e == 0] = 1

        # D_v^{-1/2}
        d_v_inv_sqrt = diags(1.0 / np.sqrt(d_v))

        # D_e^{-1}
        d_e_inv = diags(1.0 / d_e)

        # W = I (单位权重)
        W = eye(self.n_edges, format="csr")

        # L = I - D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
        # 分步计算避免内存问题
        temp = d_v_inv_sqrt @ H @ W @ d_e_inv @ H.T @ d_v_inv_sqrt
        self.L = eye(self.n_nodes, format="csr") - temp

        # 估计最大特征值（用于切比雪夫缩放）
        # 使用幂迭代法近似
        self.lambda_max = self._estimate_lambda_max()

    def _estimate_lambda_max(self, n_iter=50):
        """幂迭代法估计最大特征值"""
        x = np.random.randn(self.n_nodes)
        x = x / np.linalg.norm(x)

        for _ in range(n_iter):
            x_new = self.L @ x
            lambda_est = np.dot(x, x_new)
            norm = np.linalg.norm(x_new)
            if norm > 0:
                x = x_new / norm

        return max(lambda_est, 2.0)  # 至少为2

    def get_scaled_laplacian(self):
        """
        获取缩放后的拉普拉斯: L_tilde = 2L/lambda_max - I

        用于切比雪夫多项式展开
        """
        L_scaled = (2.0 / self.lambda_max) * self.L - eye(self.n_nodes, format="csr")
        return L_scaled

    def to_torch(self, device="cpu"):
        """转换为PyTorch稀疏张量"""
        L_scaled = self.get_scaled_laplacian()

        # 转换为COO格式
        L_coo = L_scaled.tocoo()
        indices = torch.LongTensor([L_coo.row, L_coo.col])
        values = torch.FloatTensor(L_coo.data)

        L_torch = torch.sparse_coo_tensor(
            indices, values, size=(self.n_nodes, self.n_nodes)
        ).to(device)

        return L_torch


class ChebyshevConv(nn.Module):
    """
    切比雪夫多项式近似的谱卷积

    h_out = sum_{k=0}^{K} theta_k * T_k(L_tilde) * h

    其中 T_k 是切比雪夫多项式:
    - T_0(x) = 1
    - T_1(x) = x
    - T_k(x) = 2x * T_{k-1}(x) - T_{k-2}(x)
    """

    def __init__(self, in_dim, out_dim, K=3):
        """
        Args:
            in_dim: 输入特征维度
            out_dim: 输出特征维度
            K: 切比雪夫多项式阶数
        """
        super().__init__()

        self.K = K
        self.in_dim = in_dim
        self.out_dim = out_dim

        # 每个多项式阶数的可学习系数
        self.theta = nn.Parameter(torch.randn(K + 1, in_dim, out_dim) * 0.01)

    def forward(self, x, L_tilde):
        """
        Args:
            x: [N, in_dim] 节点特征
            L_tilde: [N, N] 缩放后的拉普拉斯（稀疏张量）

        Returns:
            out: [N, out_dim] 卷积后特征
        """
        N = x.size(0)

        # 计算切比雪夫多项式基: T_0, T_1, ..., T_K
        T_list = []

        # T_0 = I => T_0 * x = x
        T_0 = x
        T_list.append(T_0)

        if self.K > 0:
            # T_1 = L_tilde => T_1 * x = L_tilde @ x
            T_1 = torch.sparse.mm(L_tilde, x)
            T_list.append(T_1)

        for k in range(2, self.K + 1):
            # T_k = 2 * L_tilde * T_{k-1} - T_{k-2}
            T_k = 2 * torch.sparse.mm(L_tilde, T_list[-1]) - T_list[-2]
            T_list.append(T_k)

        # 加权求和: sum_k theta_k * T_k
        out = torch.zeros(N, self.out_dim, device=x.device)
        for k, T_k in enumerate(T_list):
            # T_k: [N, in_dim], theta[k]: [in_dim, out_dim]
            out = out + T_k @ self.theta[k]

        return out


class SpectralHypergraphConv(nn.Module):
    """
    谱超图卷积层

    实现频率分解:
    - 低通滤波: g_low(λ) = exp(-λ/σ_low²)
    - 高通滤波: g_high(λ) = 1 - exp(-λ/σ_high²)

    通过切比雪夫多项式近似实现
    """

    def __init__(self, in_dim, out_dim, config):
        """
        Args:
            in_dim: 输入维度
            out_dim: 输出维度（h_low和h_high各占一半）
            config: 配置对象
        """
        super().__init__()

        self.freq_dim = config.freq_dim
        self.sigma_low = config.sigma_low
        self.sigma_high = config.sigma_high
        K = config.chebyshev_k

        # 低通滤波器
        self.low_pass = ChebyshevConv(in_dim, self.freq_dim, K)

        # 高通滤波器
        self.high_pass = ChebyshevConv(in_dim, self.freq_dim, K)

        # 频率融合
        self.fusion = nn.Sequential(
            nn.Linear(self.freq_dim * 2 + in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

        # 输出投影
        self.output_proj = nn.Linear(out_dim, out_dim)

    def forward(self, x, L_tilde):
        """
        Args:
            x: [N, in_dim] 节点特征
            L_tilde: [N, N] 缩放后的拉普拉斯

        Returns:
            h_fused: [N, out_dim] 融合后特征
            h_low: [N, freq_dim] 低频特征
            h_high: [N, freq_dim] 高频特征
        """
        # 低通滤波
        h_low = self.low_pass(x, L_tilde)

        # 高通滤波
        h_high = self.high_pass(x, L_tilde)

        # 融合: [h_low; h_high; x]
        h_concat = torch.cat([h_low, h_high, x], dim=-1)
        h_fused = self.fusion(h_concat)
        h_fused = self.output_proj(h_fused)

        return h_fused, h_low, h_high


class MultiLayerSpectralHypergraphEncoder(nn.Module):
    """
    多层谱超图编码器

    包含:
    1. 多层谱卷积
    2. 频率分解输出 (h_low, h_high)
    3. scGPT引导融合
    """

    def __init__(self, in_dim, config, n_layers=3):
        """
        Args:
            in_dim: 输入特征维度
            config: 配置对象
            n_layers: 谱卷积层数
        """
        super().__init__()

        self.n_layers = n_layers
        self.hidden_dim = config.tme_hidden_dim
        self.freq_dim = config.freq_dim

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )

        # 多层谱卷积
        self.spectral_layers = nn.ModuleList()
        for i in range(n_layers):
            self.spectral_layers.append(
                SpectralHypergraphConv(self.hidden_dim, self.hidden_dim, config)
            )

        # scGPT引导模块
        self.scgpt_guide = nn.ModuleDict({
            "kv_proj": nn.Linear(config.latent_dim, self.hidden_dim * 2),
            "q_proj": nn.Linear(self.hidden_dim, self.hidden_dim),
            "cross_attn": nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=4,
                dropout=config.dropout,
                batch_first=True,
            ),
            "gate": nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim), nn.Sigmoid()
            ),
            "output": nn.LayerNorm(self.hidden_dim),
        })

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, config.tme_embed_dim),
        )

        # 缓存拉普拉斯矩阵
        self.L_tilde = None

    def set_laplacian(self, L_tilde):
        """设置拉普拉斯矩阵"""
        self.L_tilde = L_tilde

    def apply_scgpt_guidance(self, h, scgpt_emb):
        """scGPT引导融合"""
        # scGPT -> K, V
        kv = self.scgpt_guide["kv_proj"](scgpt_emb)
        k, v = kv.chunk(2, dim=-1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

        # 超图表示 -> Q
        q = self.scgpt_guide["q_proj"](h).unsqueeze(1)

        # 跨注意力
        attn_out, _ = self.scgpt_guide["cross_attn"](q, k, v)
        attn_out = attn_out.squeeze(1)

        # 门控融合
        gate = self.scgpt_guide["gate"](torch.cat([h, attn_out], dim=-1))
        guided = h + gate * attn_out

        return self.scgpt_guide["output"](guided)

    def forward(self, x, center_indices, scgpt_emb=None):
        """
        Args:
            x: [N, in_dim] 全部节点特征
            center_indices: [batch] 中心节点索引
            scgpt_emb: [batch, scgpt_dim] scGPT嵌入（可选）

        Returns:
            z_tme: [batch, tme_embed_dim] TME嵌入
            h_low: [batch, freq_dim] 低频特征
            h_high: [batch, freq_dim] 高频特征
        """
        if self.L_tilde is None:
            raise ValueError("请先调用 set_laplacian() 设置拉普拉斯矩阵")

        # 输入投影
        h = self.input_proj(x)

        # 多层谱卷积
        all_h_low = []
        all_h_high = []

        for layer in self.spectral_layers:
            h, h_low, h_high = layer(h, self.L_tilde)
            all_h_low.append(h_low)
            all_h_high.append(h_high)

        # 取最后一层的频率特征
        final_h_low = all_h_low[-1]
        final_h_high = all_h_high[-1]

        # 提取中心节点表示
        center_h = h[center_indices]
        center_h_low = final_h_low[center_indices]
        center_h_high = final_h_high[center_indices]

        # scGPT引导
        if scgpt_emb is not None:
            center_h = self.apply_scgpt_guidance(center_h, scgpt_emb)

        # 输出投影
        z_tme = self.output_proj(center_h)

        return z_tme, center_h_low, center_h_high


def compute_frequency_orthogonality_loss(h_low, h_high):
    """
    计算频率正交损失 (A1配套)

    L_freq = (1/B) * sum_i cos²(h_low[i], h_high[i])

    目标：使低频和高频特征正交
    """
    # 归一化
    h_low_norm = F.normalize(h_low, dim=-1)
    h_high_norm = F.normalize(h_high, dim=-1)

    # 计算余弦相似度
    cos_sim = (h_low_norm * h_high_norm).sum(dim=-1)

    # 正交损失：最小化余弦相似度的平方
    loss = (cos_sim**2).mean()

    return loss
