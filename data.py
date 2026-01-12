#!/usr/bin/env python3
"""
数据处理模块

包含:
1. 数据加载和预处理
2. Dataset类
3. Collate函数
4. scGPT tokenization支持
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scanpy as sc
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data

from .hypergraph import LocalGraphExtractor


def load_and_preprocess_data(config, vocab=None) -> Dict[str, Any]:
    """
    加载和预处理数据

    Args:
        config: 配置对象
        vocab: scGPT词汇表（可选）

    Returns:
        data_dict: 包含所有预处理数据的字典
    """
    print("=" * 60)
    print("加载数据")
    print("=" * 60)

    # 加载h5ad文件
    data_path = os.path.join(config.data_dir, config.data_file)
    adata = sc.read_h5ad(data_path)
    print(f"总细胞数: {adata.n_obs}")
    print(f"原始基因数: {adata.n_vars}")

    # 筛选高变基因
    if adata.n_vars > config.n_genes:
        adata = _select_highly_variable_genes(adata, config.n_genes)
        print(f"筛选高变基因后: {adata.n_vars} 个基因")

    print(f"使用基因数: {adata.n_vars}")

    # 获取空间坐标（全部细胞，用于构建空间图和超边）
    spatial_coords_all = adata.obsm["spatial"]

    # 全部细胞的表达矩阵（用于TME建模）
    if hasattr(adata.X, "toarray"):
        expression_all = adata.X.toarray()
    else:
        expression_all = adata.X.copy()

    # 标准化全部细胞
    lib_size_all = expression_all.sum(axis=1, keepdims=True)
    lib_size_all[lib_size_all == 0] = 1
    expression_all = np.log1p(expression_all / lib_size_all * 10000)

    # 全部细胞的细胞类型（用于超边构建）
    cell_types_all = adata.obs["cell_type"].values if "cell_type" in adata.obs else None

    # 样本/切片标签（用于分开构建空间图）
    sample_labels_all = None
    for col in ["sample", "batch", "slice"]:
        if col in adata.obs.columns:
            sample_labels_all = adata.obs[col].values
            print(
                f"使用 '{col}' 列作为样本标签 ({len(np.unique(sample_labels_all))} 个样本)"
            )
            break

    # 筛选有扰动的细胞（用于训练）
    has_perturbation = (
        (adata.obs["perturbation"].notna())
        & (adata.obs["perturbation"] != "None")
        & (adata.obs["perturbation"] != "Multiple")
    )

    adata_perturbed = adata[has_perturbation].copy()
    print(f"有扰动的细胞数: {adata_perturbed.n_obs}")

    # 记录有扰动细胞在全部细胞中的索引
    perturbed_indices_in_all = np.where(has_perturbation.values)[0]

    # 扰动类型
    perturb_counts = adata_perturbed.obs["perturbation"].value_counts()
    perturb_names = list(perturb_counts.index)
    print(f"扰动类型数: {len(perturb_names)}")

    # 划分Control和KO
    is_control = adata_perturbed.obs["perturbation"] == "Control"
    ctrl_indices = np.where(is_control)[0]
    ko_indices = np.where(~is_control)[0]

    print(f"Control细胞数: {len(ctrl_indices)}")
    print(f"KO细胞数: {len(ko_indices)}")

    # 获取表达矩阵
    if hasattr(adata_perturbed.X, "toarray"):
        expression = adata_perturbed.X.toarray()
    else:
        expression = adata_perturbed.X

    # 标准化: log1p(TPM)
    lib_size = expression.sum(axis=1, keepdims=True)
    lib_size[lib_size == 0] = 1
    expression = np.log1p(expression / lib_size * 10000)

    # 有扰动细胞的空间坐标
    coords = spatial_coords_all[has_perturbation.values]

    # 扰动标签
    perturbations = adata_perturbed.obs["perturbation"].values
    perturb_to_idx = {name: idx for idx, name in enumerate(perturb_names)}
    perturb_labels = np.array([perturb_to_idx[p] for p in perturbations])

    # Control均值
    ctrl_mean = expression[ctrl_indices].mean(axis=0)

    # 基因名
    gene_names = (
        adata_perturbed.var["gene_name"].values
        if "gene_name" in adata_perturbed.var
        else adata_perturbed.var_names.values
    )

    # scGPT Tokenization
    gene_ids = None
    if vocab is not None:
        gene_ids = _build_gene_id_mapping(gene_names, vocab)

    # 有扰动细胞的细胞类型
    cell_types = (
        adata_perturbed.obs["cell_type"].values
        if "cell_type" in adata_perturbed.obs
        else None
    )

    return {
        # 有扰动细胞的数据（用于训练）
        "expression": expression.astype(np.float32),
        "coords": coords.astype(np.float32),
        "perturb_labels": perturb_labels,
        "perturb_names": perturb_names,
        "perturb_to_idx": perturb_to_idx,
        "ctrl_indices": ctrl_indices,
        "ko_indices": ko_indices,
        "ctrl_mean": ctrl_mean.astype(np.float32),
        "gene_names": gene_names,
        "gene_ids": gene_ids,
        "cell_types": cell_types,
        # 全部细胞的数据（用于TME建模和超边构建）
        "expression_all": expression_all.astype(np.float32),
        "coords_all": spatial_coords_all.astype(np.float32),
        "cell_types_all": cell_types_all,
        "sample_labels_all": sample_labels_all,
        "perturbed_indices_in_all": perturbed_indices_in_all,
        "n_cells_all": len(expression_all),
    }


def _select_highly_variable_genes(adata, n_genes: int):
    """选择高变基因"""
    if "highly_variable" in adata.var.columns:
        if "highly_variable_rank" in adata.var.columns:
            hvg_rank = adata.var["highly_variable_rank"].values.copy()
            hvg_rank[np.isnan(hvg_rank)] = np.inf
            top_gene_idx = np.argsort(hvg_rank)[:n_genes]
        else:
            hvg_mask = adata.var["highly_variable"].values
            hvg_indices = np.where(hvg_mask)[0]
            top_gene_idx = (
                hvg_indices[:n_genes] if len(hvg_indices) >= n_genes else hvg_indices
            )
        return adata[:, top_gene_idx].copy()
    else:
        print("计算高变基因...")
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_genes,
            flavor="seurat_v3",
            layer="counts" if "counts" in adata.layers else None,
        )
        return adata[:, adata.var["highly_variable"]].copy()


def _build_gene_id_mapping(gene_names: np.ndarray, vocab) -> np.ndarray:
    """构建基因名到vocab ID的映射"""
    print("\n构建scGPT基因映射...")
    gene_ids = []
    vocab_matched = 0

    for gene in gene_names:
        gene_upper = str(gene).upper()
        gene_lower = str(gene).lower()
        gene_orig = str(gene)

        if gene_orig in vocab:
            gene_ids.append(vocab[gene_orig])
            vocab_matched += 1
        elif gene_upper in vocab:
            gene_ids.append(vocab[gene_upper])
            vocab_matched += 1
        elif gene_lower in vocab:
            gene_ids.append(vocab[gene_lower])
            vocab_matched += 1
        else:
            gene_ids.append(-1)

    gene_ids = np.array(gene_ids, dtype=np.int64)
    print(
        f"  基因匹配率: {vocab_matched}/{len(gene_names)} ({100 * vocab_matched / len(gene_names):.1f}%)"
    )
    print(f"  有效基因数: {(gene_ids >= 0).sum()}")

    return gene_ids


class SpatialPerturbDataset(Dataset):
    """
    空间扰动预测数据集

    支持:
    - scGPT tokenization
    - 超图信息
    - 局部子图提取
    """

    def __init__(
        self,
        data_dict: Dict,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        config,
        indices: Optional[np.ndarray] = None,
        vocab=None,
        use_scgpt_input: bool = False,
        hyperedge_dict: Optional[Dict] = None,
        node_to_hyperedges: Optional[Dict] = None,
    ):
        self.expression = data_dict["expression"]
        self.coords = data_dict["coords"]
        self.perturb_labels = data_dict["perturb_labels"]
        self.ctrl_mean = data_dict["ctrl_mean"]
        self.ctrl_indices = data_dict["ctrl_indices"]
        self.n_perturbations = len(data_dict["perturb_names"])
        self.config = config

        # 全局索引映射（有扰动细胞在全部细胞中的位置）
        self.perturbed_indices_in_all = data_dict.get("perturbed_indices_in_all", None)
        self.n_cells_all = data_dict.get("n_cells_all", len(self.expression))

        # scGPT tokenization
        self.gene_ids = data_dict.get("gene_ids", None)
        self.vocab = vocab
        self.use_scgpt_input = (
            use_scgpt_input and self.gene_ids is not None and vocab is not None
        )

        if self.use_scgpt_input:
            self.valid_gene_mask = self.gene_ids >= 0
            self.valid_gene_ids = self.gene_ids[self.valid_gene_mask]
            print(
                f"  Dataset: 使用scGPT tokenization, 有效基因={self.valid_gene_mask.sum()}"
            )
        else:
            self.valid_gene_mask = None
            self.valid_gene_ids = None

        # 超边信息
        self.hyperedge_dict = hyperedge_dict
        self.node_to_hyperedges = node_to_hyperedges
        self.use_hypergraph = hyperedge_dict is not None

        if self.use_hypergraph:
            print("  Dataset: 使用超图编码")

        # 数据索引（在有扰动细胞中的索引）
        self.indices = indices if indices is not None else data_dict["ko_indices"]

        # 局部图提取器（使用全部细胞的数量）
        self.graph_extractor = LocalGraphExtractor(
            edge_index, edge_attr, self.n_cells_all, config.k_hop
        )

    def __len__(self):
        return len(self.indices)

    def _tokenize_cell(self, expr: np.ndarray) -> Tuple:
        """将细胞表达转换为scGPT输入格式"""
        if not self.use_scgpt_input:
            return None, None, None

        valid_expr = expr[self.valid_gene_mask]
        nonzero_mask = valid_expr > 0
        nonzero_indices = np.where(nonzero_mask)[0]

        if len(nonzero_indices) == 0:
            nonzero_indices = np.arange(len(self.valid_gene_ids))

        # 限制序列长度
        max_len = self.config.max_seq_len - 1
        if len(nonzero_indices) > max_len:
            expr_sorted_idx = np.argsort(valid_expr[nonzero_indices])[::-1]
            nonzero_indices = nonzero_indices[expr_sorted_idx[:max_len]]

        gene_ids = self.valid_gene_ids[nonzero_indices]
        values = valid_expr[nonzero_indices]

        # 添加<cls> token
        cls_id = self.vocab["<cls>"]
        gene_ids = np.insert(gene_ids, 0, cls_id)
        values = np.insert(values, 0, 0)

        return (torch.LongTensor(gene_ids), torch.FloatTensor(values), len(gene_ids))

    def __getitem__(self, idx):
        ko_cell_idx = self.indices[idx]

        ko_expr = self.expression[ko_cell_idx]
        perturb_idx = self.perturb_labels[ko_cell_idx]

        # 全局索引（在全部细胞中的位置）
        if self.perturbed_indices_in_all is not None:
            global_cell_idx = self.perturbed_indices_in_all[ko_cell_idx]
        else:
            global_cell_idx = ko_cell_idx

        # 随机选择一个Control细胞
        ctrl_idx = np.random.choice(self.ctrl_indices)
        ctrl_expr = self.expression[ctrl_idx]

        # 计算表达变化
        delta_x = ko_expr - self.ctrl_mean

        # 提取局部子图（使用全局索引）
        subgraph_nodes, sub_edge_index, sub_edge_attr, center_idx = (
            self.graph_extractor.extract(global_cell_idx)
        )

        result = {
            "ctrl_expr": torch.FloatTensor(ctrl_expr),
            "ko_expr": torch.FloatTensor(ko_expr),
            "delta_x": torch.FloatTensor(delta_x),
            "c_idx": perturb_idx,
            "sub_edge_index": sub_edge_index,
            "sub_edge_attr": sub_edge_attr,
            "center_idx": center_idx,
            "ko_cell_idx": ko_cell_idx,
            "global_center_idx": global_cell_idx,  # 全局索引
        }

        # scGPT tokenization
        if self.use_scgpt_input:
            gene_ids, values, seq_len = self._tokenize_cell(ctrl_expr)
            result["gene_ids"] = gene_ids
            result["values"] = values
            result["seq_len"] = seq_len

        return result


def collate_fn(batch: List[Dict], pad_token_id: int = 0, pad_value: float = 0) -> Dict:
    """
    Collate函数

    支持:
    - 图数据批处理
    - scGPT序列padding
    """
    ctrl_expr = torch.stack([b["ctrl_expr"] for b in batch])
    ko_expr = torch.stack([b["ko_expr"] for b in batch])
    delta_x = torch.stack([b["delta_x"] for b in batch])
    c_idx = torch.LongTensor([b["c_idx"] for b in batch])

    # 构建图batch（子图结构仍然可用于局部特征提取）
    graphs = []
    for b in batch:
        g = Data(
            edge_index=b["sub_edge_index"],
            edge_attr=b["sub_edge_attr"],
            center_idx=b["center_idx"],
        )
        graphs.append(g)

    batch_graph = Batch.from_data_list(graphs)

    result = {
        "ctrl_expr": ctrl_expr,
        "ko_expr": ko_expr,
        "delta_x": delta_x,
        "c_idx": c_idx,
        "graph": batch_graph,
    }

    # scGPT padding
    if "gene_ids" in batch[0] and batch[0]["gene_ids"] is not None:
        max_len = max(b["seq_len"] for b in batch)

        padded_gene_ids = []
        padded_values = []
        padding_masks = []

        for b in batch:
            seq_len = b["seq_len"]
            pad_len = max_len - seq_len

            gene_ids = b["gene_ids"]
            if pad_len > 0:
                gene_ids = torch.cat([
                    gene_ids,
                    torch.full((pad_len,), pad_token_id, dtype=torch.long),
                ])
            padded_gene_ids.append(gene_ids)

            values = b["values"]
            if pad_len > 0:
                values = torch.cat([
                    values,
                    torch.full((pad_len,), pad_value, dtype=torch.float),
                ])
            padded_values.append(values)

            mask = torch.zeros(max_len, dtype=torch.bool)
            mask[seq_len:] = True
            padding_masks.append(mask)

        result["gene_ids"] = torch.stack(padded_gene_ids)
        result["values"] = torch.stack(padded_values)
        result["padding_mask"] = torch.stack(padding_masks)

    # 全局中心节点索引（在全部细胞中的位置）
    if "global_center_idx" in batch[0]:
        result["global_center_indices"] = torch.LongTensor([
            b["global_center_idx"] for b in batch
        ])

    return result


def create_collate_fn(vocab=None):
    """创建collate函数"""
    if vocab is not None:
        try:
            pad_token_id = vocab["<pad>"]
        except KeyError:
            pad_token_id = 0
    else:
        pad_token_id = 0

    def _collate(batch):
        return collate_fn(batch, pad_token_id=pad_token_id, pad_value=0)

    return _collate


def create_dataloaders(
    data_dict: Dict,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    config,
    vocab=None,
    use_scgpt_input: bool = False,
    hyperedge_dict: Optional[Dict] = None,
    node_to_hyperedges: Optional[Dict] = None,
    train_ratio: float = 0.8,
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和测试DataLoader

    Args:
        data_dict: 数据字典
        edge_index: 边索引
        edge_attr: 边属性
        config: 配置
        vocab: scGPT词汇表
        use_scgpt_input: 是否使用scGPT输入
        hyperedge_dict: 超边字典
        node_to_hyperedges: 节点到超边映射
        train_ratio: 训练集比例

    Returns:
        train_loader, test_loader
    """
    # 划分数据
    ko_indices = data_dict["ko_indices"].copy()
    np.random.shuffle(ko_indices)

    n_train = int(len(ko_indices) * train_ratio)
    train_indices = ko_indices[:n_train]
    test_indices = ko_indices[n_train:]

    print(f"\n训练集: {len(train_indices)} 细胞")
    print(f"测试集: {len(test_indices)} 细胞")

    # 创建数据集
    train_dataset = SpatialPerturbDataset(
        data_dict,
        edge_index,
        edge_attr,
        config,
        train_indices,
        vocab=vocab,
        use_scgpt_input=use_scgpt_input,
        hyperedge_dict=hyperedge_dict,
        node_to_hyperedges=node_to_hyperedges,
    )
    test_dataset = SpatialPerturbDataset(
        data_dict,
        edge_index,
        edge_attr,
        config,
        test_indices,
        vocab=vocab,
        use_scgpt_input=use_scgpt_input,
        hyperedge_dict=hyperedge_dict,
        node_to_hyperedges=node_to_hyperedges,
    )

    # 创建collate函数
    batch_collate_fn = create_collate_fn(vocab) if use_scgpt_input else collate_fn

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=batch_collate_fn,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=batch_collate_fn,
        num_workers=0,
    )

    return train_loader, test_loader
