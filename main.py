#!/usr/bin/env python3
"""
SP-HyperRAE: Spatial Perturbation Prediction via Hypergraph-Regularized Autoencoder

主入口文件

使用方法:
    python -m sp_hyperrae.main --data GSE193460.h5ad --epochs 50

或者:
    cd sp_hyperrae
    python main.py --data GSE193460.h5ad
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sp_hyperrae.config import ModelConfig
from sp_hyperrae.data import create_dataloaders, load_and_preprocess_data
from sp_hyperrae.hypergraph import build_semantic_hyperedges, build_spatial_graph
from sp_hyperrae.model import SPHyperRAE, count_parameters
from sp_hyperrae.train import evaluate_model, print_results, save_results, train_model

# 尝试导入scGPT
try:
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab
    from scgpt.utils import load_pretrained

    SCGPT_AVAILABLE = True
    print("scGPT 加载成功!")
except ImportError:
    SCGPT_AVAILABLE = False
    print("scGPT 导入失败，将使用备用编码器")


def load_scgpt_model(config) -> tuple:
    """
    加载预训练scGPT模型

    Returns:
        scgpt_model: 预训练模型
        vocab: 基因词汇表
        use_scgpt_input: 是否使用scGPT输入格式
    """
    if not SCGPT_AVAILABLE:
        print("scGPT库未安装，使用备用编码器")
        return None, None, False

    model_dir = Path(config.scgpt_model_dir)
    if not model_dir.exists():
        print(f"scGPT模型目录不存在: {model_dir}")
        print("将使用备用编码器")
        return None, None, False

    print(f"\n尝试加载预训练scGPT: {model_dir}")

    try:
        vocab_file = model_dir / "vocab.json"
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"

        # 检查必需文件
        for f in [vocab_file, model_config_file, model_file]:
            if not f.exists():
                raise FileNotFoundError(f"缺少文件: {f}")

        # 加载vocab
        vocab = GeneVocab.from_file(vocab_file)
        print(f"  vocab加载成功，包含 {len(vocab)} 个基因")

        # 确保特殊token存在
        for token in ["<pad>", "<cls>", "<eoc>"]:
            if token not in vocab:
                vocab.append_token(token)
                print(f"  添加特殊token: {token}")

        # 加载模型配置
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        print(
            f"  模型配置: embsize={model_configs['embsize']}, "
            f"nlayers={model_configs['nlayers']}, nheads={model_configs['nheads']}"
        )

        # 创建scGPT模型
        scgpt_model = TransformerModel(
            ntoken=len(vocab),
            d_model=model_configs["embsize"],
            nhead=model_configs["nheads"],
            d_hid=model_configs["d_hid"],
            nlayers=model_configs["nlayers"],
            vocab=vocab,
            dropout=model_configs["dropout"],
            pad_token=model_configs["pad_token"],
            pad_value=model_configs["pad_value"],
            do_mvc=False,
            do_dab=False,
            use_batch_labels=False,
            input_emb_style="continuous",
            cell_emb_style="cls",
        )

        # 加载预训练权重
        load_pretrained(scgpt_model, torch.load(model_file, map_location="cpu"))
        print("  预训练scGPT加载成功!")

        # 更新配置
        config.scgpt_embed_dim = model_configs["embsize"]

        return scgpt_model, vocab, True

    except Exception as e:
        print(f"加载scGPT失败: {e}")
        print("将使用备用编码器")
        return None, None, False


def main():
    """主函数"""
    # 命令行参数
    parser = argparse.ArgumentParser(
        description="SP-HyperRAE: Spatial Perturbation Prediction via Hypergraph-Regularized Autoencoder"
    )
    parser.add_argument("--data", type=str, default=None, help="数据文件名")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=None, help="批次大小")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--n_genes", type=int, default=None, help="使用的基因数")
    parser.add_argument(
        "--use_spectral", action="store_true", help="使用谱超图卷积 (A1)"
    )
    parser.add_argument(
        "--use_contrast", action="store_true", help="使用超边对比学习 (B1+B2+B3)"
    )
    parser.add_argument(
        "--simple", action="store_true", help="使用简化版（不使用超图）"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SP-HyperRAE")
    print("Spatial Perturbation Prediction via")
    print("Hypergraph-Regularized Autoencoder")
    print("=" * 60)

    # 配置
    config = ModelConfig()

    # 命令行参数覆盖
    if args.data is not None:
        config.data_file = args.data
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr
    if args.n_genes is not None:
        config.n_genes = args.n_genes
    if args.use_spectral:
        config.use_spectral_conv = True
    if args.simple:
        config.use_spectral_conv = False
        config.use_true_hypergraph = False

    print("\n配置:")
    print(f"  数据文件: {config.data_file}")
    print(f"  训练轮数: {config.epochs}")
    print(f"  批次大小: {config.batch_size}")
    print(f"  学习率: {config.lr}")
    print(f"  基因数: {config.n_genes}")
    print(f"  使用谱超图卷积: {config.use_spectral_conv}")
    print(f"  设备: {config.device}")

    os.makedirs(config.output_dir, exist_ok=True)

    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # ========== 步骤1: 加载scGPT ==========
    scgpt_model, vocab, use_scgpt_input = load_scgpt_model(config)

    # ========== 步骤2: 加载数据 ==========
    print("\n" + "=" * 40)
    print("加载和预处理数据")
    print("=" * 40)

    data_dict = load_and_preprocess_data(config, vocab=vocab)

    # ========== 步骤3: 构建空间图（使用全部细胞，按样本分开）==========
    print(f"\n使用全部 {data_dict['n_cells_all']} 个细胞构建空间图和超边")
    edge_index, edge_attr, adj_list = build_spatial_graph(
        data_dict["coords_all"],
        k=config.k_neighbors,
        sample_labels=data_dict["sample_labels_all"],
    )

    # ========== 步骤4: 构建语义超边（使用全部细胞）==========
    hyperedge_dict = None
    node_to_hyperedges = None

    if config.use_true_hypergraph and data_dict["cell_types_all"] is not None:
        print("\n" + "=" * 40)
        print("构建语义超边")
        print("=" * 40)

        hyperedge_dict, node_to_hyperedges, _ = build_semantic_hyperedges(
            data_dict["coords_all"],
            data_dict["cell_types_all"],
            adj_list,
            config,
            sample_labels=data_dict["sample_labels_all"],
        )
    elif config.use_true_hypergraph:
        print("警告: 无法构建超边 (缺少cell_type信息)")

    # ========== 步骤5: 创建数据加载器 ==========
    train_loader, test_loader = create_dataloaders(
        data_dict,
        edge_index,
        edge_attr,
        config,
        vocab=vocab,
        use_scgpt_input=use_scgpt_input,
        hyperedge_dict=hyperedge_dict,
        node_to_hyperedges=node_to_hyperedges,
    )

    n_perturbations = len(data_dict["perturb_names"])

    # ========== 步骤6: 创建模型 ==========
    print("\n" + "=" * 40)
    print("创建 SP-HyperRAE 模型")
    print("=" * 40)

    model = SPHyperRAE(
        config.n_genes, n_perturbations, config, vocab=vocab, scgpt_model=scgpt_model
    ).to(config.device)

    # 冻结scGPT参数（在统计参数前）
    if hasattr(model, "use_scgpt") and model.use_scgpt and config.freeze_scgpt:
        print("冻结scGPT参数...")
        for param in model.cell_encoder.scgpt.parameters():
            param.requires_grad = False

    params = count_parameters(model)
    print(f"总参数量: {params['total']:,}")
    print(f"可训练参数量: {params['trainable']:,}")
    print(f"冻结参数量: {params['frozen']:,}")

    # ========== 步骤7: 训练 ==========
    print("\n" + "=" * 40)
    print("训练模型")
    print("=" * 40)

    use_full_loss = args.use_contrast or (hyperedge_dict is not None)

    model = train_model(
        model,
        train_loader,
        config,
        use_scgpt_input=use_scgpt_input,
        hyperedge_dict=hyperedge_dict,
        node_to_hyperedges=node_to_hyperedges,
        all_expression=data_dict["expression_all"],  # 使用全部细胞
        use_full_loss=use_full_loss,
    )

    # ========== 步骤8: 评估 ==========
    print("\n" + "=" * 40)
    print("评估模型")
    print("=" * 40)

    results = evaluate_model(
        model,
        test_loader,
        data_dict,
        config,
        hyperedge_dict=hyperedge_dict,
        node_to_hyperedges=node_to_hyperedges,
        all_expression=data_dict["expression_all"],  # 使用全部细胞
    )

    # ========== 步骤9: 输出结果 ==========
    print_results(results)

    # ========== 步骤10: 保存 ==========
    save_results(model, results, config, config.output_dir)


def print_usage():
    """打印使用说明"""
    print("""
============================================================
                    SP-HyperRAE 使用说明
============================================================

1. 基础运行:
    python -m sp_hyperrae.main --data GSE193460.h5ad

2. 使用谱超图卷积 (A1):
    python -m sp_hyperrae.main --data GSE193460.h5ad --use_spectral

3. 使用超边对比学习 (B1+B2+B3):
    python -m sp_hyperrae.main --data GSE193460.h5ad --use_contrast

4. 完整功能:
    python -m sp_hyperrae.main --data GSE193460.h5ad --use_spectral --use_contrast --epochs 100

5. 简化版（不使用超图）:
    python -m sp_hyperrae.main --data GSE193460.h5ad --simple

配置scGPT:
- 下载预训练模型到 scGPT/save/scGPT_human/
- 需要的文件: vocab.json, args.json, best_model.pt

============================================================
""")


if __name__ == "__main__":
    main()
