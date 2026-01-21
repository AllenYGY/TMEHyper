#!/usr/bin/env python3
"""
超边构建模块

构建四种语义超边:
1. T_contact: T细胞接触超边（免疫浸润区域）
2. Tumor_contact: 肿瘤细胞接触超边（肿瘤侵袭前沿）
3. Interface: 界面超边（肿瘤-免疫交界区）
4. Spatial: 空间邻近超边（基于K近邻）

每种超边编码不同的生物学信息，帮助模型学习微环境结构
"""

from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from scipy.sparse import coo_matrix
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors, kneighbors_graph


def build_spatial_graph(
    coords: np.ndarray, k: int = 15, sample_labels: np.ndarray = None
) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]]]:
    """
    构建空间K近邻图

    Args:
        coords: [N, 2] 空间坐标
        k: 近邻数量
        sample_labels: [N] 样本/切片标签，不同样本的细胞不会互相连接

    Returns:
        edge_index: [2, E] 边索引
        edge_attr: [E, 3] 边属性 (距离, dx, dy)
        adj_list: 邻接表
    """
    n_cells = len(coords)
    adj_list = [[] for _ in range(n_cells)]
    all_edges = []
    all_attrs = []

    if sample_labels is None:
        # 没有样本标签，所有细胞一起构建
        samples = [np.arange(n_cells)]
        sample_names = ["all"]
    else:
        # 按样本分开构建
        unique_samples = np.unique(sample_labels)
        samples = [np.where(sample_labels == s)[0] for s in unique_samples]
        sample_names = unique_samples
        print(f"\n按样本分开构建空间图 ({len(unique_samples)} 个样本)...")

    for sample_idx, (sample_indices, sample_name) in enumerate(
        zip(samples, sample_names)
    ):
        if len(sample_indices) < k + 1:
            print(f"  样本 {sample_name}: 细胞数 {len(sample_indices)} < k+1，跳过")
            continue

        sample_coords = coords[sample_indices]
        actual_k = min(k, len(sample_indices) - 1)

        # 该样本内的K近邻图
        adj = kneighbors_graph(
            sample_coords, actual_k, mode="connectivity", include_self=False
        )
        adj = adj + adj.T  # 对称化
        adj = (adj > 0).astype(float)

        coo = coo_matrix(adj)

        # 转换为全局索引
        global_src = sample_indices[coo.row]
        global_dst = sample_indices[coo.col]

        # 计算边属性
        diff = coords[global_dst] - coords[global_src]
        distances = np.linalg.norm(diff, axis=1, keepdims=True)
        distances[distances == 0] = 1e-6
        directions = diff / distances

        # 归一化距离（样本内）
        if distances.max() > 0:
            distances = distances / distances.max()

        edge_attr = np.hstack([distances, directions]).astype(np.float32)

        all_edges.append(np.array([global_src, global_dst]))
        all_attrs.append(edge_attr)

        # 更新邻接表
        for s, d in zip(global_src, global_dst):
            adj_list[s].append(d)

    # 合并所有边
    if all_edges:
        edge_index = np.hstack(all_edges)
        edge_attr = np.vstack(all_attrs)
    else:
        edge_index = np.array([[], []], dtype=np.int64)
        edge_attr = np.array([]).reshape(0, 3).astype(np.float32)

    print(f"总边数: {edge_index.shape[1]}")

    return torch.LongTensor(edge_index), torch.FloatTensor(edge_attr), adj_list


def estimate_dbscan_eps(coords: np.ndarray, k: int = 15, percentile: int = 70) -> float:
    """
    自动估计DBSCAN的eps参数

    基于K近邻距离来估计合适的eps值

    Args:
        coords: 坐标
        k: 用于估计的近邻数
        percentile: 使用第几百分位的距离

    Returns:
        eps: 估计的eps值
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    # 取第k个邻居的距离作为eps
    k_distances = distances[:, -1]
    eps = np.percentile(k_distances, percentile)

    # 添加最小eps阈值：基于坐标范围的2%
    coord_range = np.max(coords, axis=0) - np.min(coords, axis=0)
    min_eps = np.mean(coord_range) * 0.02
    eps = max(eps, min_eps)

    return eps


def infer_cell_type_mapping(cell_types: np.ndarray) -> Dict[str, int]:
    """
    自动推断细胞类型映射

    根据细胞类型名称中的关键词来推断是T细胞还是肿瘤细胞

    Args:
        cell_types: 细胞类型数组

    Returns:
        cell_type_map: 细胞类型到类别的映射
    """
    unique_types = np.unique(cell_types)
    cell_type_map = {}

    # T细胞关键词
    t_keywords = [
        "T_cell",
        "T cell",
        "Tcell",
        "T-cell",
        "CD4",
        "CD8",
        "T_",
        "immune",
        "lymphocyte",
    ]
    # 肿瘤细胞关键词
    tumor_keywords = [
        "Tumor",
        "tumor",
        "Cancer",
        "cancer",
        "Malignant",
        "malignant",
        "Epithelial",
    ]
    # 重叠/交界关键词
    overlap_keywords = ["overlap", "interface", "boundary", "mixed"]

    print("\n细胞类型映射:")
    for ct in unique_types:
        ct_str = str(ct)
        ct_lower = ct_str.lower()

        # 检查是否是T细胞
        is_t = any(kw.lower() in ct_lower for kw in t_keywords)
        # 检查是否是肿瘤细胞
        is_tumor = any(kw.lower() in ct_lower for kw in tumor_keywords)
        # 检查是否是重叠区
        is_overlap = any(kw.lower() in ct_lower for kw in overlap_keywords)

        if is_overlap or (is_t and is_tumor):
            cell_type_map[ct_str] = 2  # Overlap
            print(f"  {ct_str} -> Overlap (2)")
        elif is_t:
            cell_type_map[ct_str] = 0  # T cell
            print(f"  {ct_str} -> T_cell (0)")
        elif is_tumor:
            cell_type_map[ct_str] = 1  # Tumor
            print(f"  {ct_str} -> Tumor (1)")
        else:
            cell_type_map[ct_str] = 3  # Background/Other
            print(f"  {ct_str} -> Background (3)")

    return cell_type_map


def build_semantic_hyperedges(
    coords: np.ndarray,
    cell_types: np.ndarray,
    adj_list: List[List[int]],
    config,
    auto_eps: bool = True,
    sample_labels: np.ndarray = None,
) -> Tuple[Dict[str, List[Set[int]]], Dict[int, List[Tuple[str, int]]], Dict]:
    """
    构建语义超边

    四种超边类型:
    1. T_contact: 按T细胞接触比例分层 + DBSCAN聚类
    2. Tumor_contact: 按肿瘤细胞接触比例分层 + DBSCAN聚类
    3. Interface: 同时接触T细胞和肿瘤细胞的界面区域
    4. Spatial: 基于空间邻近的超边

    Args:
        coords: [N, 2] 空间坐标
        cell_types: [N] 细胞类型标签
        adj_list: 邻接表
        config: 配置对象
        auto_eps: 是否自动估计eps

    Returns:
        hyperedge_dict: {type: [set of node indices, ...]}
        node_to_hyperedges: {node_idx: [(type, edge_idx), ...]}
        neighborhood_features: {feature_name: np.array}
    """
    n_cells = len(coords)

    print("\n构建语义超边...")
    print(f"总细胞数: {n_cells}")

    # 自动推断细胞类型映射
    cell_type_map = infer_cell_type_mapping(cell_types)
    ct_labels = np.array([cell_type_map.get(str(ct), 3) for ct in cell_types])

    # 统计细胞类型分布
    n_t_cells = (ct_labels == 0).sum()
    n_tumor_cells = (ct_labels == 1).sum()
    n_overlap = (ct_labels == 2).sum()
    n_background = (ct_labels == 3).sum()

    print("\n细胞类型分布:")
    print(f"  T_cell (0): {n_t_cells}")
    print(f"  Tumor (1): {n_tumor_cells}")
    print(f"  Overlap (2): {n_overlap}")
    print(f"  Background (3): {n_background}")

    # 检查是否只有单一细胞类型
    if n_t_cells == 0 and n_tumor_cells > 0:
        print("\n⚠️  警告: 数据中没有T细胞!")
        print("   -> T_contact 和 Interface 超边将无法构建")
        print("   -> 模型将主要依赖 Tumor_contact 和 Spatial 超边")
    elif n_tumor_cells == 0 and n_t_cells > 0:
        print("\n⚠️  警告: 数据中没有肿瘤细胞!")
        print("   -> Tumor_contact 和 Interface 超边将无法构建")
        print("   -> 模型将主要依赖 T_contact 和 Spatial 超边")
    elif n_t_cells == 0 and n_tumor_cells == 0:
        print("\n⚠️  警告: 数据中没有T细胞和肿瘤细胞!")
        print("   -> 所有语义超边将无法构建，仅使用 Spatial 超边")

    # 计算每个细胞的邻域特征
    t_ratios = np.zeros(n_cells)
    tumor_ratios = np.zeros(n_cells)
    is_interface = np.zeros(n_cells)

    for i in range(n_cells):
        neighbors = adj_list[i]
        if len(neighbors) > 0:
            neighbor_types = ct_labels[neighbors]
            t_ratios[i] = (neighbor_types == 0).mean()
            tumor_ratios[i] = (neighbor_types == 1).mean()
            has_t = (neighbor_types == 0).any()
            has_tumor = (neighbor_types == 1).any()
            is_interface[i] = float(has_t and has_tumor)

    # 打印邻域特征统计
    print("\n邻域特征统计:")
    print(f"  T接触比例 > 0: {(t_ratios > 0).sum()} 细胞")
    print(f"  Tumor接触比例 > 0: {(tumor_ratios > 0).sum()} 细胞")
    print(f"  Interface细胞: {(is_interface > 0).sum()} 细胞")

    # 初始化超边字典
    hyperedge_dict = {
        "t_contact": [],
        "tumor_contact": [],
        "interface": [],
        "spatial": [],
    }
    node_to_hyperedges = {i: [] for i in range(n_cells)}

    # 超边参数 - 降低min_samples以适应稀疏数据
    min_samples = getattr(config, "hyperedge_min_samples", 3)

    # 自动估计eps或使用配置
    if auto_eps:
        eps = estimate_dbscan_eps(coords, k=15, percentile=70)
        print(f"\n自动估计 DBSCAN eps: {eps:.2f}")
    else:
        eps = getattr(config, "hyperedge_eps", 100)
        print(f"\n使用配置 DBSCAN eps: {eps}")

    # 使用更粗的bins分层，减少数据分割
    bins = [0, 0.01, 0.1, 0.3, 1.0]

    # ========== 1. T Cell Contact超边 ==========
    print("\n构建 T_contact 超边...")
    _build_ratio_based_hyperedges(
        coords,
        t_ratios,
        bins,
        eps,
        min_samples,
        hyperedge_dict,
        node_to_hyperedges,
        "t_contact",
        sample_labels=sample_labels,
    )

    # ========== 2. Tumor Contact超边 ==========
    print("构建 Tumor_contact 超边...")
    _build_ratio_based_hyperedges(
        coords,
        tumor_ratios,
        bins,
        eps,
        min_samples,
        hyperedge_dict,
        node_to_hyperedges,
        "tumor_contact",
        sample_labels=sample_labels,
    )

    # ========== 3. Interface超边 ==========
    print("构建 Interface 超边...")
    _build_interface_hyperedges(
        coords,
        is_interface,
        eps,
        min_samples,
        hyperedge_dict,
        node_to_hyperedges,
        sample_labels=sample_labels,
    )

    # ========== 4. Spatial超边 (K近邻) ==========
    # 对于大数据集，跳过spatial超边（空间关系已被K近邻图捕获）
    max_cells_for_spatial = getattr(config, "max_cells_for_spatial_hyperedge", 50000)
    if n_cells <= max_cells_for_spatial:
        print("构建 Spatial 超边...")
        _build_spatial_hyperedges(
            adj_list, hyperedge_dict, node_to_hyperedges, min_neighbors=3
        )
    else:
        print(
            f"跳过 Spatial 超边 (细胞数 {n_cells} > {max_cells_for_spatial}，使用语义超边)"
        )
        # 对大数据集，可以采样构建少量spatial超边
        _build_sampled_spatial_hyperedges(
            adj_list,
            hyperedge_dict,
            node_to_hyperedges,
            n_samples=min(1000, n_cells // 100),
            min_neighbors=3,
        )

    # 统计
    print("\n超边构建完成:")
    for etype, edges in hyperedge_dict.items():
        if len(edges) > 0:
            avg_size = np.mean([len(e) for e in edges])
            print(f"  {etype}: {len(edges)} 超边, 平均大小: {avg_size:.1f}")
        else:
            print(f"  {etype}: {len(edges)} 超边")

    neighborhood_features = {
        "t_ratios": t_ratios,
        "tumor_ratios": tumor_ratios,
        "is_interface": is_interface,
    }

    return hyperedge_dict, node_to_hyperedges, neighborhood_features


def _build_ratio_based_hyperedges(
    coords: np.ndarray,
    ratios: np.ndarray,
    bins: List[float],
    eps: float,
    min_samples: int,
    hyperedge_dict: Dict,
    node_to_hyperedges: Dict,
    edge_type: str,
    min_ratio_threshold: float = 0.01,
    sample_labels: np.ndarray = None,
):
    """
    按比例分层构建超边（按样本分开聚类，带回退机制）

    Args:
        min_ratio_threshold: 最小比例阈值，低于此值的细胞不参与构建
        sample_labels: 样本标签，不同样本分开聚类
    """
    total_edges = 0

    # 只考虑有实际接触的细胞 (ratio > threshold)
    valid_mask = ratios >= min_ratio_threshold
    n_valid = valid_mask.sum()

    if n_valid == 0:
        print(f"    无有效细胞 (ratio >= {min_ratio_threshold}), 跳过")
        return

    # 确定样本列表
    if sample_labels is None:
        unique_samples = [None]
    else:
        unique_samples = np.unique(sample_labels)

    for i in range(len(bins) - 1):
        # 跳过第一个bin如果它低于阈值
        if bins[i + 1] <= min_ratio_threshold:
            continue

        # 调整bin的下界
        low = max(bins[i], min_ratio_threshold)
        high = bins[i + 1]

        if i == len(bins) - 2:
            bin_mask = ratios >= low
        else:
            bin_mask = (ratios >= low) & (ratios < high)

        bin_total_clusters = 0
        n_bin_total = bin_mask.sum()

        # 按样本分开聚类
        for sample in unique_samples:
            if sample is None:
                sample_mask = bin_mask
            else:
                sample_mask = bin_mask & (sample_labels == sample)

            n_in_bin = sample_mask.sum()
            if n_in_bin < min_samples:
                continue

            bin_indices = np.where(sample_mask)[0]
            bin_coords = coords[sample_mask]

            # DBSCAN聚类（样本内）
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(bin_coords)
            n_clusters = (
                max(clustering.labels_) + 1
                if len(set(clustering.labels_)) > 1 or clustering.labels_[0] != -1
                else 0
            )

            for c in range(n_clusters):
                c_mask = clustering.labels_ == c
                if c_mask.sum() >= min_samples:
                    edge_nodes = set(bin_indices[c_mask].tolist())
                    edge_idx = len(hyperedge_dict[edge_type])
                    hyperedge_dict[edge_type].append(edge_nodes)
                    for node in edge_nodes:
                        node_to_hyperedges[node].append((edge_type, edge_idx))
                    total_edges += 1
                    bin_total_clusters += 1

        # 回退机制：如果按样本聚类失败，尝试整体聚类（忽略样本边界）
        if bin_total_clusters == 0 and n_bin_total >= min_samples:
            bin_indices = np.where(bin_mask)[0]
            bin_coords = coords[bin_mask]

            # 尝试更大的eps
            larger_eps = eps * 2.0
            clustering = DBSCAN(eps=larger_eps, min_samples=min_samples).fit(bin_coords)
            n_clusters = (
                max(clustering.labels_) + 1
                if len(set(clustering.labels_)) > 1 or clustering.labels_[0] != -1
                else 0
            )

            for c in range(n_clusters):
                c_mask = clustering.labels_ == c
                if c_mask.sum() >= min_samples:
                    edge_nodes = set(bin_indices[c_mask].tolist())
                    edge_idx = len(hyperedge_dict[edge_type])
                    hyperedge_dict[edge_type].append(edge_nodes)
                    for node in edge_nodes:
                        node_to_hyperedges[node].append((edge_type, edge_idx))
                    total_edges += 1
                    bin_total_clusters += 1

            if bin_total_clusters > 0:
                print(
                    f"    bin [{low:.2f}, {high:.2f}): {n_bin_total} 细胞 -> {bin_total_clusters} 簇 (回退模式)"
                )

        # 调试信息
        if bin_total_clusters > 0 and n_bin_total > 0:
            print(
                f"    bin [{low:.2f}, {high:.2f}): {n_bin_total} 细胞 -> {bin_total_clusters} 簇"
            )

    # 最终回退：如果仍然没有超边，基于K近邻构建
    if total_edges == 0 and n_valid >= min_samples:
        print("    DBSCAN失败，使用K近邻回退...")
        valid_indices = np.where(valid_mask)[0]
        valid_coords = coords[valid_mask]

        # 使用K近邻构建超边
        k = min(15, n_valid - 1)
        if k >= min_samples:
            nbrs = NearestNeighbors(n_neighbors=k).fit(valid_coords)
            _, indices = nbrs.kneighbors(valid_coords)

            # 每隔一定数量的细胞创建一个超边
            step = max(1, n_valid // 50)  # 最多创建50个超边
            for idx in range(0, n_valid, step):
                neighbor_local_indices = indices[idx]
                edge_nodes = set(valid_indices[neighbor_local_indices].tolist())
                if len(edge_nodes) >= min_samples:
                    edge_idx = len(hyperedge_dict[edge_type])
                    hyperedge_dict[edge_type].append(edge_nodes)
                    for node in edge_nodes:
                        node_to_hyperedges[node].append((edge_type, edge_idx))
                    total_edges += 1

        print(f"    K近邻回退: {total_edges} 超边")

    if total_edges == 0:
        print(f"    未能构建任何超边 (有效细胞: {n_valid})")


def _build_interface_hyperedges(
    coords: np.ndarray,
    is_interface: np.ndarray,
    eps: float,
    min_samples: int,
    hyperedge_dict: Dict,
    node_to_hyperedges: Dict,
    sample_labels: np.ndarray = None,
):
    """构建界面超边（按样本分开聚类，带回退机制）"""
    interface_mask = is_interface > 0.5
    n_interface = interface_mask.sum()

    if n_interface < min_samples:
        print(f"    Interface: {n_interface} 细胞 (不足 {min_samples})")
        return

    # 确定样本列表
    if sample_labels is None:
        unique_samples = [None]
    else:
        unique_samples = np.unique(sample_labels)

    total_clusters = 0

    for sample in unique_samples:
        if sample is None:
            sample_mask = interface_mask
        else:
            sample_mask = interface_mask & (sample_labels == sample)

        n_in_sample = sample_mask.sum()
        if n_in_sample < min_samples:
            continue

        sample_indices = np.where(sample_mask)[0]
        sample_coords = coords[sample_mask]

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(sample_coords)
        n_clusters = (
            max(clustering.labels_) + 1
            if len(set(clustering.labels_)) > 1 or clustering.labels_[0] != -1
            else 0
        )

        for c in range(n_clusters):
            c_mask = clustering.labels_ == c
            if c_mask.sum() >= min_samples:
                edge_nodes = set(sample_indices[c_mask].tolist())
                edge_idx = len(hyperedge_dict["interface"])
                hyperedge_dict["interface"].append(edge_nodes)
                for node in edge_nodes:
                    node_to_hyperedges[node].append(("interface", edge_idx))
                total_clusters += 1

    # 回退机制：如果按样本聚类失败，尝试整体聚类
    if total_clusters == 0 and n_interface >= min_samples:
        interface_indices = np.where(interface_mask)[0]
        interface_coords = coords[interface_mask]

        # 尝试更大的eps
        larger_eps = eps * 2.0
        clustering = DBSCAN(eps=larger_eps, min_samples=min_samples).fit(
            interface_coords
        )
        n_clusters = (
            max(clustering.labels_) + 1
            if len(set(clustering.labels_)) > 1 or clustering.labels_[0] != -1
            else 0
        )

        for c in range(n_clusters):
            c_mask = clustering.labels_ == c
            if c_mask.sum() >= min_samples:
                edge_nodes = set(interface_indices[c_mask].tolist())
                edge_idx = len(hyperedge_dict["interface"])
                hyperedge_dict["interface"].append(edge_nodes)
                for node in edge_nodes:
                    node_to_hyperedges[node].append(("interface", edge_idx))
                total_clusters += 1

        if total_clusters > 0:
            print(
                f"    Interface: {n_interface} 细胞 -> {total_clusters} 簇 (回退模式)"
            )

    # 最终回退：使用K近邻
    if total_clusters == 0 and n_interface >= min_samples:
        interface_indices = np.where(interface_mask)[0]
        interface_coords = coords[interface_mask]

        k = min(15, n_interface - 1)
        if k >= min_samples:
            nbrs = NearestNeighbors(n_neighbors=k).fit(interface_coords)
            _, indices = nbrs.kneighbors(interface_coords)

            step = max(1, n_interface // 20)
            for idx in range(0, n_interface, step):
                neighbor_local_indices = indices[idx]
                edge_nodes = set(interface_indices[neighbor_local_indices].tolist())
                if len(edge_nodes) >= min_samples:
                    edge_idx = len(hyperedge_dict["interface"])
                    hyperedge_dict["interface"].append(edge_nodes)
                    for node in edge_nodes:
                        node_to_hyperedges[node].append(("interface", edge_idx))
                    total_clusters += 1

        print(f"    Interface: {n_interface} 细胞 -> {total_clusters} 簇 (K近邻回退)")

    if total_clusters == 0:
        print(f"    Interface: {n_interface} 细胞 -> 0 簇")
    elif "回退" not in str(total_clusters):
        print(f"    Interface: {n_interface} 细胞 -> {total_clusters} 簇")


def _build_spatial_hyperedges(
    adj_list: List[List[int]],
    hyperedge_dict: Dict,
    node_to_hyperedges: Dict,
    min_neighbors: int = 3,
):
    """构建空间邻近超边"""
    n_cells = len(adj_list)
    for i in range(n_cells):
        neighbors = adj_list[i]
        if len(neighbors) >= min_neighbors:
            edge_nodes = set([i] + list(neighbors))
            edge_idx = len(hyperedge_dict["spatial"])
            hyperedge_dict["spatial"].append(edge_nodes)
            for node in edge_nodes:
                node_to_hyperedges[node].append(("spatial", edge_idx))


def _build_sampled_spatial_hyperedges(
    adj_list: List[List[int]],
    hyperedge_dict: Dict,
    node_to_hyperedges: Dict,
    n_samples: int = 1000,
    min_neighbors: int = 3,
):
    """构建采样的空间邻近超边（用于大数据集）"""
    n_cells = len(adj_list)

    # 均匀采样细胞索引
    if n_samples >= n_cells:
        sample_indices = range(n_cells)
    else:
        sample_indices = np.random.choice(n_cells, n_samples, replace=False)

    count = 0
    for i in sample_indices:
        neighbors = adj_list[i]
        if len(neighbors) >= min_neighbors:
            edge_nodes = set([i] + list(neighbors))
            edge_idx = len(hyperedge_dict["spatial"])
            hyperedge_dict["spatial"].append(edge_nodes)
            for node in edge_nodes:
                node_to_hyperedges[node].append(("spatial", edge_idx))
            count += 1

    print(
        f"    采样 Spatial 超边: {count} 个 (从 {n_cells} 细胞中采样 {len(sample_indices)} 个)"
    )


class LocalGraphExtractor:
    """提取K-hop局部子图"""

    def __init__(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_nodes: int,
        k_hop: int = 2,
    ):
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.num_nodes = num_nodes
        self.k_hop = k_hop
        self._build_adjacency_list()

    def _build_adjacency_list(self):
        """构建邻接表和边字典"""
        self.adj_list = [[] for _ in range(self.num_nodes)]
        self.edge_dict = {}

        src, dst = self.edge_index.numpy()
        for idx, (s, d) in enumerate(zip(src, dst)):
            self.adj_list[s].append(d)
            self.edge_dict[(s, d)] = idx

    def extract(self, center_idx: int) -> Tuple:
        """
        提取以center_idx为中心的K-hop子图

        Args:
            center_idx: 中心节点索引

        Returns:
            subgraph_nodes: 子图节点列表
            sub_edge_index: 子图边索引
            sub_edge_attr: 子图边属性
            center_in_subgraph: 中心节点在子图中的索引
        """
        # BFS扩展K-hop
        visited = {center_idx}
        frontier = [center_idx]

        for _ in range(self.k_hop):
            next_frontier = []
            for node in frontier:
                for neighbor in self.adj_list[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier

        # 构建子图
        subgraph_nodes = sorted(list(visited))
        node_map = {old: new for new, old in enumerate(subgraph_nodes)}
        center_in_subgraph = node_map[center_idx]

        # 收集边
        sub_edges = []
        sub_attrs = []

        for s in subgraph_nodes:
            for d in self.adj_list[s]:
                if d in visited:
                    edge_idx = self.edge_dict.get((s, d))
                    if edge_idx is not None:
                        sub_edges.append([node_map[s], node_map[d]])
                        sub_attrs.append(self.edge_attr[edge_idx].numpy())

        # 处理空边情况
        if len(sub_edges) == 0:
            sub_edge_index = torch.LongTensor([[0], [0]])
            sub_edge_attr = torch.zeros(1, self.edge_attr.shape[1])
        else:
            sub_edge_index = torch.LongTensor(sub_edges).T
            sub_edge_attr = torch.FloatTensor(np.array(sub_attrs))

        return subgraph_nodes, sub_edge_index, sub_edge_attr, center_in_subgraph


def get_hyperedge_statistics(
    hyperedge_dict: Dict[str, List[Set[int]]],
    node_to_hyperedges: Dict[int, List[Tuple[str, int]]],
) -> Dict:
    """
    获取超边统计信息

    Args:
        hyperedge_dict: 超边字典
        node_to_hyperedges: 节点到超边映射

    Returns:
        stats: 统计信息字典
    """
    stats = {
        "n_hyperedges": {},
        "avg_edge_size": {},
        "avg_node_degree": 0,
    }

    total_edges = 0
    for etype, edges in hyperedge_dict.items():
        n_edges = len(edges)
        stats["n_hyperedges"][etype] = n_edges
        total_edges += n_edges

        if n_edges > 0:
            avg_size = np.mean([len(e) for e in edges])
            stats["avg_edge_size"][etype] = avg_size

    # 平均节点度（每个节点属于多少个超边）
    degrees = [len(v) for v in node_to_hyperedges.values()]
    stats["avg_node_degree"] = np.mean(degrees) if degrees else 0
    stats["total_hyperedges"] = total_edges

    return stats
