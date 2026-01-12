#!/usr/bin/env python3
"""
超边对比学习模块 (B1 + B2 + B3)

B1: 超边内聚合损失 - 同一超边内的节点表示应该相近
B2: 超边类型对比损失 - 同类型超边相似，不同类型超边区分
B3: 超边间分离损失 - 不同超边保持最小距离（防止表示坍塌）

生物学意义:
- B1: 同一微环境内的细胞应该有相似的状态表示
- B2: 不同类型微环境（T_contact, Tumor_contact, Interface）有不同的特征
- B3: 每个微环境都应该有独特的表示，不会坍塌成相同值
"""

from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperedgeContrastiveLoss(nn.Module):
    """
    超边对比学习损失

    包含三个子损失:
    - L_intra (B1): 超边内聚合
    - L_type (B2): 类型对比
    - L_inter (B3): 超边分离
    """

    def __init__(self, config):
        """
        Args:
            config: 配置对象，包含损失权重等参数
        """
        super().__init__()

        self.intra_weight = config.intra_weight
        self.type_weight = config.type_weight
        self.inter_weight = config.inter_weight
        self.margin = config.inter_margin
        self.temperature = config.contrast_temperature

    def compute_hyperedge_embeddings(
        self,
        z_nodes: torch.Tensor,
        hyperedge_dict: Dict[str, List[Set[int]]],
        device: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """
        计算超边嵌入（聚合超边内节点的表示）

        z_edge[j] = mean(z_nodes[nodes in edge j])

        Args:
            z_nodes: [N, dim] 所有节点的表示
            hyperedge_dict: {type: [set of node indices, ...]}
            device: 设备

        Returns:
            edge_embeddings: {type: [n_edges, dim] 超边嵌入}
        """
        edge_embeddings = {}

        for etype, edges in hyperedge_dict.items():
            if len(edges) == 0:
                continue

            edge_embs = []
            for edge_nodes in edges:
                if len(edge_nodes) == 0:
                    continue
                node_indices = list(edge_nodes)
                # 确保索引在有效范围内
                valid_indices = [i for i in node_indices if i < z_nodes.size(0)]
                if len(valid_indices) == 0:
                    continue
                # 均值聚合
                edge_emb = z_nodes[valid_indices].mean(dim=0)
                edge_embs.append(edge_emb)

            if len(edge_embs) > 0:
                edge_embeddings[etype] = torch.stack(edge_embs)

        return edge_embeddings

    def intra_hyperedge_loss(
        self,
        z_nodes: torch.Tensor,
        hyperedge_dict: Dict[str, List[Set[int]]],
        max_edges_per_type: int = 100,
    ) -> torch.Tensor:
        """
        B1: 超边内聚合损失

        L_intra = (1/|E|) * sum_j (1/|E_j|) * sum_{i in E_j} ||z_i - z_bar_j||²

        目标：同一超边内的节点表示应该接近超边中心

        Args:
            z_nodes: [N, dim] 节点表示
            hyperedge_dict: 超边字典
            max_edges_per_type: 每种类型采样的最大超边数

        Returns:
            loss: 标量损失
        """
        total_loss = 0.0
        n_edges = 0

        for etype, edges in hyperedge_dict.items():
            if len(edges) == 0:
                continue

            # 采样超边（如果太多）
            if len(edges) > max_edges_per_type:
                sampled_indices = torch.randperm(len(edges))[:max_edges_per_type]
                sampled_edges = [edges[i] for i in sampled_indices]
            else:
                sampled_edges = edges

            for edge_nodes in sampled_edges:
                if len(edge_nodes) < 2:
                    continue

                node_indices = list(edge_nodes)
                valid_indices = [i for i in node_indices if i < z_nodes.size(0)]
                if len(valid_indices) < 2:
                    continue

                # 超边内节点的表示
                node_embs = z_nodes[valid_indices]

                # 超边中心
                edge_center = node_embs.mean(dim=0, keepdim=True)

                # 节点到中心的距离
                distances = (node_embs - edge_center).pow(2).sum(dim=-1)

                # 平均距离
                edge_loss = distances.mean()
                total_loss = total_loss + edge_loss
                n_edges += 1

        if n_edges == 0:
            return torch.tensor(0.0, device=z_nodes.device)

        return total_loss / n_edges

    def type_contrastive_loss(
        self, edge_embeddings: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        B2: 超边类型对比损失 (InfoNCE)

        L_type = -(1/|E|) * sum_j log(
            sum_{k in P(j)} exp(sim(z_j, z_k) / tau) /
            sum_{k != j} exp(sim(z_j, z_k) / tau)
        )

        其中 P(j) = {k: type(k) == type(j), k != j} 是同类型正样本集

        目标：同类型超边相似，不同类型超边区分

        Args:
            edge_embeddings: {type: [n_edges, dim]}

        Returns:
            loss: 标量损失
        """
        # 收集所有超边嵌入和类型标签
        all_embs = []
        all_types = []

        type_to_idx = {}
        for idx, (etype, embs) in enumerate(edge_embeddings.items()):
            type_to_idx[etype] = idx
            n_edges = embs.size(0)
            all_embs.append(embs)
            all_types.extend([idx] * n_edges)

        if len(all_embs) == 0:
            return torch.tensor(0.0)

        all_embs = torch.cat(all_embs, dim=0)  # [total_edges, dim]
        all_types = torch.tensor(all_types, device=all_embs.device)

        n_total = all_embs.size(0)
        if n_total < 2:
            return torch.tensor(0.0, device=all_embs.device)

        # 归一化
        all_embs_norm = F.normalize(all_embs, dim=-1)

        # 计算所有对的相似度
        sim_matrix = torch.mm(all_embs_norm, all_embs_norm.t())  # [n, n]
        sim_matrix = sim_matrix / self.temperature

        # 创建类型掩码：同类型为正样本
        type_mask = all_types.unsqueeze(0) == all_types.unsqueeze(1)  # [n, n]

        # 排除自身
        self_mask = torch.eye(n_total, dtype=torch.bool, device=all_embs.device)
        type_mask = type_mask & (~self_mask)

        # 检查是否有正样本
        has_positive = type_mask.sum(dim=1) > 0

        if has_positive.sum() == 0:
            return torch.tensor(0.0, device=all_embs.device)

        # InfoNCE损失
        # 分母：所有非自身样本
        exp_sim = torch.exp(sim_matrix)
        exp_sim = exp_sim * (~self_mask).float()
        denominator = exp_sim.sum(dim=1)

        # 分子：同类型样本
        exp_sim_pos = exp_sim * type_mask.float()
        numerator = exp_sim_pos.sum(dim=1)

        # 防止除零
        numerator = numerator + 1e-8
        denominator = denominator + 1e-8

        # 只对有正样本的计算损失
        loss = -torch.log(numerator / denominator)
        loss = loss[has_positive].mean()

        return loss

    def inter_hyperedge_loss(
        self, edge_embeddings: Dict[str, torch.Tensor], max_pairs: int = 500
    ) -> torch.Tensor:
        """
        B3: 超边间分离损失 (margin-based)

        L_inter = (1/|E|(|E|-1)) * sum_{j != k} max(0, m - ||z_j - z_k||)²

        目标：不同超边保持最小距离m，防止表示坍塌

        Args:
            edge_embeddings: {type: [n_edges, dim]}
            max_pairs: 最大采样对数

        Returns:
            loss: 标量损失
        """
        # 收集所有超边嵌入
        all_embs = []
        for embs in edge_embeddings.values():
            all_embs.append(embs)

        if len(all_embs) == 0:
            return torch.tensor(0.0)

        all_embs = torch.cat(all_embs, dim=0)  # [total_edges, dim]
        n_total = all_embs.size(0)

        if n_total < 2:
            return torch.tensor(0.0, device=all_embs.device)

        # 计算所有对的距离
        # 使用采样来避免O(n²)复杂度
        n_pairs = n_total * (n_total - 1) // 2

        if n_pairs > max_pairs:
            # 随机采样对
            idx1 = torch.randint(0, n_total, (max_pairs,), device=all_embs.device)
            idx2 = torch.randint(0, n_total, (max_pairs,), device=all_embs.device)
            # 确保不是同一个
            same_mask = idx1 == idx2
            idx2[same_mask] = (idx2[same_mask] + 1) % n_total
        else:
            # 使用所有对
            idx1 = []
            idx2 = []
            for i in range(n_total):
                for j in range(i + 1, n_total):
                    idx1.append(i)
                    idx2.append(j)
            idx1 = torch.tensor(idx1, device=all_embs.device)
            idx2 = torch.tensor(idx2, device=all_embs.device)

        # 计算距离
        emb1 = all_embs[idx1]
        emb2 = all_embs[idx2]
        distances = (emb1 - emb2).norm(dim=-1)

        # Margin损失
        margin_violation = F.relu(self.margin - distances)
        loss = (margin_violation**2).mean()

        return loss

    def forward(
        self,
        z_nodes: torch.Tensor,
        hyperedge_dict: Dict[str, List[Set[int]]],
        batch_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算所有超边对比学习损失

        Args:
            z_nodes: [N, dim] 节点表示
            hyperedge_dict: 超边字典
            batch_indices: [batch] 当前batch的节点索引（可选）

        Returns:
            losses: {
                'intra': B1损失,
                'type': B2损失,
                'inter': B3损失,
                'total': 加权总损失
            }
        """
        device = z_nodes.device
        losses = {}

        # 计算超边嵌入
        edge_embeddings = self.compute_hyperedge_embeddings(
            z_nodes, hyperedge_dict, device
        )

        # B1: 超边内聚合
        losses["intra"] = self.intra_hyperedge_loss(z_nodes, hyperedge_dict)

        # B2: 类型对比
        losses["type"] = self.type_contrastive_loss(edge_embeddings)

        # B3: 超边分离
        losses["inter"] = self.inter_hyperedge_loss(edge_embeddings)

        # 加权总损失
        losses["total"] = (
            self.intra_weight * losses["intra"]
            + self.type_weight * losses["type"]
            + self.inter_weight * losses["inter"]
        )

        return losses


class HyperedgeEmbeddingExtractor(nn.Module):
    """
    超边嵌入提取器

    用于输出z_edge供下游分析
    """

    def __init__(self, node_dim: int, edge_dim: int):
        """
        Args:
            node_dim: 节点表示维度
            edge_dim: 超边嵌入维度
        """
        super().__init__()

        self.aggregator = nn.Sequential(
            nn.Linear(node_dim, edge_dim),
            nn.LayerNorm(edge_dim),
            nn.GELU(),
        )

    def forward(
        self, z_nodes: torch.Tensor, hyperedge_dict: Dict[str, List[Set[int]]]
    ) -> Dict[str, Dict]:
        """
        提取所有超边的嵌入

        Args:
            z_nodes: [N, dim] 节点表示
            hyperedge_dict: 超边字典

        Returns:
            hyperedge_features: {
                'type_name': {
                    'z_edge': [n_edges, edge_dim],
                    'node_indices': [[nodes], ...]
                },
                ...
            }
        """
        hyperedge_features = {}

        for etype, edges in hyperedge_dict.items():
            if len(edges) == 0:
                continue

            edge_embs = []
            node_indices_list = []

            for edge_nodes in edges:
                if len(edge_nodes) == 0:
                    continue

                node_indices = list(edge_nodes)
                valid_indices = [i for i in node_indices if i < z_nodes.size(0)]

                if len(valid_indices) == 0:
                    continue

                # 聚合节点表示
                node_embs = z_nodes[valid_indices]
                edge_emb = node_embs.mean(dim=0)
                edge_emb = self.aggregator(edge_emb.unsqueeze(0)).squeeze(0)

                edge_embs.append(edge_emb)
                node_indices_list.append(valid_indices)

            if len(edge_embs) > 0:
                hyperedge_features[etype] = {
                    "z_edge": torch.stack(edge_embs),
                    "node_indices": node_indices_list,
                }

        return hyperedge_features
