#!/usr/bin/env python3
"""
训练和评估模块

包含:
1. 训练循环
2. 评估函数
3. 结果可视化
"""

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import SimpleLoss, SPHyperRAELoss


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    config,
    use_scgpt_input: bool = False,
    hyperedge_dict: Optional[Dict] = None,
    node_to_hyperedges: Optional[Dict] = None,
    all_expression: Optional[np.ndarray] = None,
    use_full_loss: bool = True,
) -> nn.Module:
    """
    训练模型

    Args:
        model: SP-HyperRAE模型
        train_loader: 训练数据加载器
        config: 配置对象
        use_scgpt_input: 是否使用scGPT输入
        hyperedge_dict: 超边字典
        node_to_hyperedges: 节点到超边映射
        all_expression: 完整表达矩阵
        use_full_loss: 是否使用完整损失（包含对比损失）

    Returns:
        训练后的模型
    """
    print("\n训练参数:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Device: {config.device}")
    print(f"  使用scGPT输入: {use_scgpt_input}")
    print(f"  使用超图编码: {hyperedge_dict is not None}")
    print(f"  使用完整损失: {use_full_loss}")

    # 设置超图结构
    if hyperedge_dict is not None and hasattr(model, "set_hypergraph"):
        n_nodes = (
            len(all_expression)
            if all_expression is not None
            else train_loader.dataset.expression.shape[0]
        )
        model.set_hypergraph(hyperedge_dict, node_to_hyperedges, n_nodes, config.device)

    # 处理scGPT参数
    scgpt_params = []
    other_params = []

    if hasattr(model, "use_scgpt") and model.use_scgpt:
        if config.freeze_scgpt:
            # 确保scGPT参数已冻结（可能在main.py中已冻结）
            frozen_count = 0
            for param in model.cell_encoder.scgpt.parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_count += 1
            if frozen_count > 0:
                print(f"  冻结了 {frozen_count} 个scGPT参数")
            else:
                print("  scGPT参数已冻结")
        else:
            _setup_scgpt_finetuning(model, config, scgpt_params)

    # 收集其他参数
    scgpt_param_ids = set(id(p) for p in scgpt_params)
    for param in model.parameters():
        if param.requires_grad and id(param) not in scgpt_param_ids:
            other_params.append(param)

    # 优化器
    if scgpt_params:
        print(
            f"  scGPT参数: {sum(p.numel() for p in scgpt_params):,}, lr={config.lr * config.scgpt_lr_scale}"
        )
        print(f"  其他参数: {sum(p.numel() for p in other_params):,}, lr={config.lr}")
        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "lr": config.lr},
                {"params": scgpt_params, "lr": config.lr * config.scgpt_lr_scale},
            ],
            weight_decay=config.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    # 损失函数
    loss_fn = SPHyperRAELoss(config) if use_full_loss else SimpleLoss(config)

    model.train()
    best_loss = float("inf")
    history = []

    for epoch in range(config.epochs):
        total_losses = {}

        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}", leave=False
        )

        for batch in pbar:
            # 准备数据
            ctrl_expr = batch["ctrl_expr"].to(config.device)
            delta_x = batch["delta_x"].to(config.device)
            c_idx = batch["c_idx"].to(config.device)

            # scGPT输入
            gene_ids = batch.get("gene_ids")
            values = batch.get("values")
            padding_mask = batch.get("padding_mask")

            if gene_ids is not None:
                gene_ids = gene_ids.to(config.device)
                values = values.to(config.device)
                padding_mask = padding_mask.to(config.device)

            # 全局中心索引
            global_center_indices = batch.get("global_center_indices")
            if global_center_indices is not None:
                global_center_indices = global_center_indices.to(config.device)

            # 前向传播
            output = model(
                ctrl_expr,
                c_idx,
                gene_ids=gene_ids,
                values=values,
                padding_mask=padding_mask,
                global_center_indices=global_center_indices,
                all_expression=all_expression,
            )

            # 计算损失
            if use_full_loss:
                # 获取全局节点表示用于对比损失
                z_nodes = output.get("z_tme")

                losses = loss_fn(
                    pred=output["delta_x_pred"],
                    target=delta_x,
                    mu=output.get("mu"),
                    logvar=output.get("logvar"),
                    z_tme=output.get("z_tme"),
                    h_low=output.get("h_low"),
                    h_high=output.get("h_high"),
                    z_nodes=z_nodes,
                    hyperedge_dict=hyperedge_dict,
                )
            else:
                losses = loss_fn(
                    pred=output["delta_x_pred"],
                    target=delta_x,
                    mu=output.get("mu"),
                    logvar=output.get("logvar"),
                    z_tme=output.get("z_tme"),
                )

            loss = losses["total"]

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # 累计损失
            for k, v in losses.items():
                if k not in total_losses:
                    total_losses[k] = 0
                total_losses[k] += v.item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        # 计算平均损失
        n_batches = len(train_loader)
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        history.append(avg_losses)

        # 打印
        loss_str = (
            f"Epoch {epoch + 1:3d}/{config.epochs} | Loss: {avg_losses['total']:.4f}"
        )
        if "recon" in avg_losses:
            loss_str += f" | Recon: {avg_losses['recon']:.4f}"
        if "freq" in avg_losses and avg_losses["freq"] > 0:
            loss_str += f" | Freq: {avg_losses['freq']:.4f}"
        print(loss_str)

        if avg_losses["total"] < best_loss:
            best_loss = avg_losses["total"]

    return model


def _setup_scgpt_finetuning(model, config, scgpt_params):
    """设置scGPT微调"""
    scgpt_model = model.cell_encoder.scgpt
    n_layers = len(scgpt_model.transformer_encoder.layers)

    if config.scgpt_finetune_layers == -1:
        print(f"  微调全部scGPT参数 (lr_scale={config.scgpt_lr_scale})")
        for param in scgpt_model.parameters():
            param.requires_grad = True
        scgpt_params.extend(list(scgpt_model.parameters()))
    elif config.scgpt_finetune_layers == 0:
        print("  冻结全部scGPT参数")
        for param in scgpt_model.parameters():
            param.requires_grad = False
    else:
        freeze_layers = n_layers - config.scgpt_finetune_layers
        print(
            f"  scGPT: 冻结前{freeze_layers}层，微调后{config.scgpt_finetune_layers}层"
        )

        for param in scgpt_model.encoder.parameters():
            param.requires_grad = False
        for param in scgpt_model.value_encoder.parameters():
            param.requires_grad = False

        for i, layer in enumerate(scgpt_model.transformer_encoder.layers):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True
                    scgpt_params.append(param)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    data_dict: Dict,
    config,
    hyperedge_dict: Optional[Dict] = None,
    node_to_hyperedges: Optional[Dict] = None,
    all_expression: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """
    评估模型

    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        data_dict: 数据字典
        config: 配置对象
        hyperedge_dict: 超边字典
        node_to_hyperedges: 节点到超边映射
        all_expression: 完整表达矩阵

    Returns:
        results: {perturb_name: {metric: value, ...}, ...}
    """
    model.eval()

    perturb_names = data_dict["perturb_names"]

    all_preds = []
    all_trues = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            ctrl_expr = batch["ctrl_expr"].to(config.device)
            delta_x = batch["delta_x"].to(config.device)
            c_idx = batch["c_idx"].to(config.device)

            # scGPT输入
            gene_ids = batch.get("gene_ids")
            values = batch.get("values")
            padding_mask = batch.get("padding_mask")

            if gene_ids is not None:
                gene_ids = gene_ids.to(config.device)
                values = values.to(config.device)
                padding_mask = padding_mask.to(config.device)

            # 全局中心索引
            global_center_indices = batch.get("global_center_indices")
            if global_center_indices is not None:
                global_center_indices = global_center_indices.to(config.device)

            # 前向传播
            output = model(
                ctrl_expr,
                c_idx,
                gene_ids=gene_ids,
                values=values,
                padding_mask=padding_mask,
                global_center_indices=global_center_indices,
                all_expression=all_expression,
            )

            delta_pred = output["delta_x_pred"]

            all_preds.append(delta_pred.cpu().numpy())
            all_trues.append(delta_x.cpu().numpy())
            all_labels.append(batch["c_idx"].numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    results = {}

    for perturb_idx, perturb_name in enumerate(perturb_names):
        if perturb_name == "Control":
            continue

        mask = all_labels == perturb_idx
        if mask.sum() < 5:
            continue

        pred_delta = all_preds[mask]
        true_delta = all_trues[mask]

        mean_pred = pred_delta.mean(axis=0)
        mean_true = true_delta.mean(axis=0)

        # 均值级评估
        valid = np.isfinite(mean_pred) & np.isfinite(mean_true)
        if valid.sum() > 2:
            delta_pearson = pearsonr(mean_pred[valid], mean_true[valid])[0]
            delta_spearman = spearmanr(mean_pred[valid], mean_true[valid])[0]
        else:
            delta_pearson = delta_spearman = np.nan

        mse_all = np.mean((mean_pred - mean_true) ** 2)

        # DE基因评估 (Top 20)
        de_indices_20 = np.argsort(np.abs(mean_true))[-20:]
        if len(de_indices_20) > 2:
            pearson_top20 = pearsonr(
                mean_pred[de_indices_20], mean_true[de_indices_20]
            )[0]
        else:
            pearson_top20 = np.nan

        pred_de_20 = set(np.argsort(np.abs(mean_pred))[-20:])
        overlap_20 = len(set(de_indices_20) & pred_de_20) / 20

        results[perturb_name] = {
            "Delta_Pearson": delta_pearson,
            "Delta_Spearman": delta_spearman,
            "MSE": mse_all,
            "Pearson_top20_DE": pearson_top20,
            "Overlap_top20": overlap_20,
            "n_cells": mask.sum(),
        }

    return results


def print_results(results: Dict[str, Dict[str, float]]):
    """打印评估结果"""
    df = pd.DataFrame(results).T

    print("\n" + "=" * 60)
    print("  SP-HyperRAE 评估结果")
    print("=" * 60)

    print("\n【均值级评估】")
    for m in ["Delta_Pearson", "Delta_Spearman", "MSE"]:
        if m in df.columns:
            print(f"  {m}: {df[m].mean():.4f} ± {df[m].std():.4f}")

    print("\n【DE基因评估】")
    for m in ["Pearson_top20_DE", "Overlap_top20"]:
        if m in df.columns:
            print(f"  {m}: {df[m].mean():.4f} ± {df[m].std():.4f}")

    print(f"\n评估了 {len(df)} 种扰动类型")

    return df


def save_results(model: nn.Module, results: Dict, config, output_dir: str):
    """保存结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存评估结果
    df = pd.DataFrame(results).T
    df.to_csv(f"{output_dir}/sp_hyperrae_results.csv")

    # 保存模型
    torch.save(model.state_dict(), f"{output_dir}/sp_hyperrae.pt")

    # 保存配置
    config_dict = {
        k: v
        for k, v in config.__class__.__dict__.items()
        if not k.startswith("_") and not callable(v)
    }
    pd.Series(config_dict).to_json(f"{output_dir}/config.json")

    print(f"\n结果保存在: {output_dir}")
