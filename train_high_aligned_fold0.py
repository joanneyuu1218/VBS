from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


PROCESS_WEIGHTS = {
    "WWjj_EW_LL_WW_cmf": 18.29,
    "WWjj_EW_LT_WW_cmf": 58.88,
    "WWjj_EW_TT_WW_cmf": 124.50,
    "WWjj_EW_LL_pp_cmf": 11.49,
    "WWjj_EW_LT_pp_cmf": 67.84,
    "WWjj_EW_TT_pp_cmf": 123.07,
    "WWjj_EW": 206.52,
    "WWjj_QCD": 24.05,
    "WZjj_EW": 14.95,
    "WZjj_QCD": 28.50,
}

TYPE_ID = {
    "padding": 0,
    "met": 1,
    "lepton": 2,
    "jet": 3,
}

VARIANT_CONFIGS = {
    "aligned": {
        "name": "senior_aligned_high_level",
        "embed_dim": 64,
        "continuous_embed_dims": [64, 256, 48],
        "type_embed_dim": 16,
        "num_heads": 4,
        "particle_fc_dim": 256,
        "class_fc_dim": 256,
        "particle_dropout": 0.1,
        "class_dropout": 0.0,
        "num_particle_blocks": 3,
        "num_class_blocks": 1,
    },
    "half": {
        "name": "half_embedding_and_blocks_high_level",
        "embed_dim": 32,
        "continuous_embed_dims": [32, 128, 24],
        "type_embed_dim": 8,
        "num_heads": 4,
        "particle_fc_dim": 128,
        "class_fc_dim": 128,
        "particle_dropout": 0.1,
        "class_dropout": 0.0,
        "num_particle_blocks": 2,
        "num_class_blocks": 1,
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wrap_phi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def read_core_sequence(df: pd.DataFrame, n_obj: int = 5) -> Tuple[np.ndarray, ...]:
    n = len(df)
    pt = np.zeros((n, n_obj), dtype=np.float32)
    eta = np.zeros_like(pt)
    phi = np.zeros_like(pt)
    mass = np.zeros_like(pt)
    typ = np.zeros((n, n_obj), dtype=np.int64)
    mask = np.zeros((n, n_obj), dtype=np.float32)
    for i in range(n_obj):
        pt[:, i] = np.maximum(df["core%d_pt" % i].to_numpy(np.float32), 0.0)
        eta[:, i] = df["core%d_eta" % i].to_numpy(np.float32)
        phi[:, i] = df["core%d_phi" % i].to_numpy(np.float32)
        mass[:, i] = np.maximum(df["core%d_m" % i].to_numpy(np.float32), 0.0)
        typ[:, i] = df["core%d_type" % i].to_numpy(np.int64)
        mask[:, i] = df["core%d_mask" % i].to_numpy(np.float32)
    return pt, eta, phi, mass, typ, mask


def build_high_tensors(df: pd.DataFrame, pt_log_scale: bool, pt_floor: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    pt, eta, phi, mass, typ, mask = read_core_sequence(df, 5)
    pt_feature = np.log10(np.maximum(pt, pt_floor)) if pt_log_scale else pt
    mass_feature = np.log10(np.maximum(mass, pt_floor))
    phi_rel = wrap_phi(phi - phi[:, [0]])
    nodes = np.stack([pt_feature, eta, phi_rel, mass_feature], axis=-1).astype(np.float32)
    nodes *= mask[:, :, None]

    d_eta = eta[:, :, None] - eta[:, None, :]
    d_phi = wrap_phi(phi[:, :, None] - phi[:, None, :])
    d_r = np.sqrt(d_eta**2 + d_phi**2)
    edges = np.stack([d_eta, d_phi, d_r], axis=-1).astype(np.float32)
    edges *= mask[:, :, None, None] * mask[:, None, :, None]

    manifest = {
        "model_level": "high-level",
        "input_objects": ["leading lepton", "subleading lepton", "leading tagging jet", "subleading tagging jet", "MET"],
        "node_features": ["log10(pt)" if pt_log_scale else "pt", "eta", "delta_phi_to_leading_lepton", "log10(mass)", "type_embedding"],
        "edge_features": ["delta_eta", "delta_phi", "delta_R"],
        "global_features": [],
        "node_shape_per_event": [int(nodes.shape[1]), int(nodes.shape[2])],
        "type_shape_per_event": [int(typ.shape[1])],
        "edge_shape_per_event": [int(edges.shape[1]), int(edges.shape[2]), int(edges.shape[3])],
        "edge_calculation": "calculated during training from core eta and core phi; not stored as parquet columns",
        "pt_log_scale": bool(pt_log_scale),
        "mass_floor": float(pt_floor),
        "embedding_note": "continuous node features are embedded with an MLP and concatenated with a learned type embedding, matching the previous v4 feature definition.",
    }
    return nodes, edges, mask.astype(np.float32), typ, manifest


def apply_hybrid_weights(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Weight"] = out["Weight"] * out["Process"].map(PROCESS_WEIGHTS).fillna(1.0)
    out["Train_Weight"] = np.abs(out["Weight"])
    sig = out.loc[out["Label"] == 1, "Train_Weight"].sum()
    bkg = out.loc[out["Label"] == 0, "Train_Weight"].sum()
    if sig > 0 and bkg > 0:
        out.loc[out["Label"] == 1, "Train_Weight"] *= bkg / sig
    return out


class HighDataset(Dataset):
    def __init__(self, x, c, v, mask, y, w):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.c = torch.tensor(c, dtype=torch.long)
        self.v = torch.tensor(v, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        self.w = torch.tensor(w, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.c[i], self.v[i], self.mask[i], self.y[i], self.w[i]


class MLPEmbed(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i != len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ParticleAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, fc_dim: int, dropout: float):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("embed dim must be divisible by num_heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, fc_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(fc_dim, dim), nn.Dropout(dropout))

    def forward(self, x, edge_bias, key_mask):
        b, n, dim = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn + edge_bias
        attn = attn.masked_fill(~key_mask[:, None, None, :], -1e4)
        attn = torch.softmax(attn, dim=-1)
        h = (self.drop(attn) @ v).transpose(1, 2).reshape(b, n, dim)
        x = x + self.drop(self.proj(h))
        return x + self.mlp(self.norm2(x))


class ClassAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, fc_dim: int, dropout: float):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("embed dim must be divisible by num_heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.norm1 = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, fc_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(fc_dim, dim), nn.Dropout(dropout))

    def forward(self, cls_token, x, particle_mask):
        b, n, dim = x.shape
        tokens = torch.cat([cls_token, x], dim=1)
        full_mask = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=x.device), particle_mask], dim=1)
        q = self.q(self.norm1(cls_token)).reshape(b, 1, self.heads, self.head_dim).transpose(1, 2)
        kv = self.kv(self.norm1(tokens)).reshape(b, n + 1, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(~full_mask[:, None, None, :], -1e4)
        attn = torch.softmax(attn, dim=-1)
        h = (self.drop(attn) @ v).transpose(1, 2).reshape(b, 1, dim)
        cls_token = cls_token + self.drop(self.proj(h))
        return cls_token + self.mlp(self.norm2(cls_token))


class AlignedHighLevelParT(nn.Module):
    def __init__(self, input_dim: int, edge_dim: int, n_types: int, cfg: Dict[str, Any], score_dim: int = 1):
        super().__init__()
        dim = int(cfg["embed_dim"])
        heads = int(cfg["num_heads"])
        type_dim = int(cfg["type_embed_dim"])
        continuous_dims = list(cfg["continuous_embed_dims"])
        if continuous_dims[-1] + type_dim != dim:
            raise ValueError("continuous embedding dim + type embedding dim must equal embed_dim")
        self.node_embed = MLPEmbed([input_dim] + continuous_dims)
        self.type_embed = nn.Embedding(max(int(n_types), 8) + 1, type_dim)
        self.edge = nn.Sequential(nn.Linear(edge_dim, 32), nn.GELU(), nn.Linear(32, heads))
        self.particle_blocks = nn.ModuleList(
            [ParticleAttentionBlock(dim, heads, int(cfg["particle_fc_dim"]), float(cfg["particle_dropout"])) for _ in range(int(cfg["num_particle_blocks"]))]
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.class_blocks = nn.ModuleList(
            [ClassAttentionBlock(dim, heads, int(cfg["class_fc_dim"]), float(cfg["class_dropout"])) for _ in range(int(cfg["num_class_blocks"]))]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim, score_dim), nn.Sigmoid())

    def forward(self, x, c, edge_features, mask):
        b, n, _ = x.shape
        x = torch.cat([self.node_embed(x), self.type_embed(c)], dim=-1)
        edge_bias = self.edge(edge_features).permute(0, 3, 1, 2)
        for block in self.particle_blocks:
            x = block(x, edge_bias, mask)
        cls = self.cls_token.expand(b, -1, -1)
        for block in self.class_blocks:
            cls = block(cls, x, mask)
        cls = self.norm(cls[:, 0])
        return self.head(cls)


@dataclass
class Split:
    x: np.ndarray
    c: np.ndarray
    v: np.ndarray
    mask: np.ndarray
    y: np.ndarray
    w: np.ndarray


def safe_auc(y, p, w):
    try:
        return float(roc_auc_score(y, p, sample_weight=w))
    except ValueError:
        return 0.5


def evaluate(model, loader, crit, device):
    model.eval()
    total_loss_weighted = 0.0
    total_weight = 0.0
    y_true, y_pred, weights = [], [], []
    with torch.no_grad():
        for xb, cb, vb, mb, yb, wb in loader:
            xb, cb, vb, mb = xb.to(device), cb.to(device), vb.to(device), mb.to(device)
            yb, wb = yb.to(device), wb.to(device)
            pred = model(xb, cb, vb, mb)
            batch_loss = crit(pred, yb)
            total_loss_weighted += float((batch_loss * wb).sum().cpu())
            total_weight += float(wb.sum().cpu())
            y_true.extend(yb.cpu().numpy().ravel())
            y_pred.extend(pred.cpu().numpy().ravel())
            weights.extend(wb.cpu().numpy().ravel())
    return float(total_loss_weighted / max(total_weight, 1e-12)), safe_auc(y_true, y_pred, weights)


def load_state_dict_compat(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def smooth_values(values: List[float], window: int = 3) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < window:
        return arr
    smoothed = np.zeros_like(arr)
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        smoothed[i] = arr[lo : i + 1].mean()
    return smoothed


def plot_history(history: Dict[str, Any], out_prefix: str):
    fold = history["fold0"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    epochs = fold["epoch"]
    best_epoch = fold.get("best_epoch")
    axes[0].plot(epochs, fold["train_loss"], color="tab:blue", alpha=0.30, linewidth=1.2, label="train raw")
    axes[0].plot(epochs, fold["val_loss"], color="tab:orange", alpha=0.30, linewidth=1.2, linestyle="--", label="validation raw")
    axes[0].plot(epochs, smooth_values(fold["train_loss"]), color="tab:blue", linewidth=2.0, label="train smoothed")
    axes[0].plot(epochs, smooth_values(fold["val_loss"]), color="tab:orange", linewidth=2.0, linestyle="--", label="validation smoothed")
    if best_epoch is not None:
        axes[0].axvline(best_epoch, color="black", linestyle=":", linewidth=1.2, label="selected epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Weighted BCE loss")
    axes[0].set_title("Fold 0 loss")
    axes[1].plot(epochs, fold["train_auc"], color="tab:blue", alpha=0.30, linewidth=1.2, label="train raw")
    axes[1].plot(epochs, fold["val_auc"], color="tab:orange", alpha=0.30, linewidth=1.2, linestyle="--", label="validation raw")
    axes[1].plot(epochs, smooth_values(fold["train_auc"]), color="tab:blue", linewidth=2.0, label="train smoothed")
    axes[1].plot(epochs, smooth_values(fold["val_auc"]), color="tab:orange", linewidth=2.0, linestyle="--", label="validation smoothed")
    if best_epoch is not None:
        axes[1].axvline(best_epoch, color="black", linestyle=":", linewidth=1.2, label="selected epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Weighted ROC AUC")
    axes[1].set_title("Fold 0 AUC")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_prefix + "_training_curves.pdf")
    plt.close(fig)


def make_loader(split: Split, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(HighDataset(split.x, split.c, split.v, split.mask, split.y, split.w), batch_size=batch_size, shuffle=shuffle)


def train(args):
    set_seed(args.seed)
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    cfg = dict(VARIANT_CONFIGS[args.variant])

    df = apply_hybrid_weights(pd.read_parquet(args.input))
    if "Fold_ID" not in df.columns:
        df["Fold_ID"] = np.arange(len(df)) % args.folds

    x, v, mask, typ, feature_manifest = build_high_tensors(df, args.pt_log_scale, args.pt_floor)
    y = df["Label"].to_numpy(np.float32)
    w = df["Train_Weight"].to_numpy(np.float32)

    fold = int(args.fold)
    test = df["Fold_ID"].to_numpy() == fold
    val = df["Fold_ID"].to_numpy() == ((fold + 1) % args.folds)
    train_mask = ~(test | val)

    sx = StandardScaler().fit(x[train_mask].reshape(-1, x.shape[-1]))
    sv = StandardScaler().fit(v[train_mask].reshape(-1, v.shape[-1]))

    def make(sel):
        b = int(sel.sum())
        return Split(
            sx.transform(x[sel].reshape(-1, x.shape[-1])).reshape(b, x.shape[1], x.shape[2]),
            typ[sel],
            sv.transform(v[sel].reshape(-1, v.shape[-1])).reshape(b, v.shape[1], v.shape[2], v.shape[3]),
            mask[sel],
            y[sel],
            w[sel],
        )

    tr, va, te = make(train_mask), make(val), make(test)
    train_loader = make_loader(tr, args.batch_size, True)
    val_loader = make_loader(va, args.batch_size, False)
    test_loader = make_loader(te, args.batch_size, False)

    model = AlignedHighLevelParT(x.shape[-1], v.shape[-1], int(typ.max()), cfg, score_dim=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max" if args.selection_metric == "val_auc" else "min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )
    crit = nn.BCELoss(reduction="none")
    if args.selection_metric == "val_auc":
        best_metric = -float("inf")
    else:
        best_metric = float("inf")
    patience_count = 0
    best_path = "%s_%s_fold%d.pt" % (args.output_prefix, args.variant, fold)
    prep_path = "%s_%s_fold%d_preprocess.pkl" % (args.output_prefix, args.variant, fold)

    with open(prep_path, "wb") as handle:
        pickle.dump(
            {
                "scaler_x": sx,
                "scaler_v": sv,
                "variant": args.variant,
                "model_config": cfg,
                "input_dim": int(x.shape[-1]),
                "edge_dim": int(v.shape[-1]),
                "n_types": int(typ.max()),
                "pt_log_scale": bool(args.pt_log_scale),
                "mass_floor": float(args.pt_floor),
            },
            handle,
        )

    fold_hist = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_auc": [],
        "val_auc": [],
        "lr": [],
        "best_epoch": None,
        "selection_metric": args.selection_metric,
        "best_selection_metric_value": None,
        "test_auc": None,
        "split_counts": {"train": int(train_mask.sum()), "validation": int(val.sum()), "test": int(test.sum())},
        "split_weight_sums": {"train": float(w[train_mask].sum()), "validation": float(w[val].sum()), "test": float(w[test].sum())},
    }

    for epoch in range(args.epochs):
        model.train()
        for xb, cb, vb, mb, yb, wb in train_loader:
            xb, cb, vb, mb = xb.to(device), cb.to(device), vb.to(device), mb.to(device)
            yb, wb = yb.to(device), wb.to(device)
            optimizer.zero_grad()
            pred = model(xb, cb, vb, mb)
            loss = (crit(pred, yb) * (wb / wb.mean())).mean()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        train_loss, train_auc = evaluate(model, train_loader, crit, device)
        val_loss, val_auc = evaluate(model, val_loader, crit, device)
        fold_hist["epoch"].append(epoch + 1)
        fold_hist["train_loss"].append(train_loss)
        fold_hist["val_loss"].append(val_loss)
        fold_hist["train_auc"].append(train_auc)
        fold_hist["val_auc"].append(val_auc)
        fold_hist["lr"].append(float(optimizer.param_groups[0]["lr"]))
        print(
            "fold %d epoch %03d train_loss=%.5f val_loss=%.5f train_auc=%.5f val_auc=%.5f lr=%.3g"
            % (fold, epoch + 1, train_loss, val_loss, train_auc, val_auc, optimizer.param_groups[0]["lr"]),
            flush=True,
        )

        if args.selection_metric == "val_auc":
            current_metric = val_auc
            improved = current_metric > best_metric + args.min_delta
        else:
            current_metric = val_loss
            improved = current_metric < best_metric - args.min_delta

        if improved:
            best_metric = current_metric
            patience_count = 0
            fold_hist["best_epoch"] = epoch + 1
            fold_hist["best_selection_metric_value"] = float(current_metric)
            torch.save(model.state_dict(), best_path)
        else:
            patience_count += 1
        if scheduler is not None:
            scheduler.step(current_metric)
        if patience_count >= args.patience:
            break

    model.load_state_dict(load_state_dict_compat(best_path, device))
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, cb, vb, mb, _, _ in test_loader:
            preds.extend(model(xb.to(device), cb.to(device), vb.to(device), mb.to(device)).cpu().numpy().ravel())
    preds = np.asarray(preds, dtype=np.float32)
    df[args.score_column] = np.nan
    df.loc[test, args.score_column] = preds
    fold_hist["test_auc"] = safe_auc(y[test], preds, w[test])

    elapsed = time.time() - start
    history = {
        "script": os.path.abspath(__file__),
        "input": args.input,
        "variant": args.variant,
        "fold": fold,
        "fold0": fold_hist,
        "elapsed_seconds": float(elapsed),
        "elapsed_minutes": float(elapsed / 60.0),
        "model_config": cfg,
        "feature_manifest": feature_manifest,
        "training_config": {
            "optimizer": "Adam",
            "learning_rate": float(args.lr),
            "batch_size": int(args.batch_size),
            "max_epochs": int(args.epochs),
            "early_stopping_patience": int(args.patience),
            "seed": int(args.seed),
            "gradient_clipping_max_norm": float(args.grad_clip),
            "selection_metric": args.selection_metric,
            "selection_min_delta": float(args.min_delta),
            "lr_scheduler": args.lr_scheduler,
            "lr_factor": float(args.lr_factor),
            "lr_patience": int(args.lr_patience),
            "min_lr": float(args.min_lr),
            "weight_strategy": "Hybrid: Weight *= process factor; Train_Weight=abs(Weight); signal Train_Weight rescaled to match background sum.",
        },
        "weighting": {
            "process_weights": PROCESS_WEIGHTS,
            "train_weight_label_sums": {str(k): float(vv) for k, vv in df.groupby("Label")["Train_Weight"].sum().to_dict().items()},
            "weighted_process_sums": {str(k): float(vv) for k, vv in df.groupby("Process")["Weight"].sum().to_dict().items()},
        },
    }

    scored_path = "%s_%s_scored.parquet" % (args.output_prefix, args.variant)
    hist_path = "%s_%s_history.json" % (args.output_prefix, args.variant)
    df.to_parquet(scored_path)
    with open(hist_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    plot_history(history, "%s_%s" % (args.output_prefix, args.variant))
    print("fold %d test_auc=%.6f" % (fold, fold_hist["test_auc"]))
    print("elapsed_seconds=%.2f" % elapsed)
    print("wrote %s" % scored_path)
    print("wrote %s" % hist_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--variant", choices=sorted(VARIANT_CONFIGS), default="aligned")
    parser.add_argument("--score-column", default="aligned_part_score")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--selection-metric", choices=["val_loss", "val_auc"], default="val_loss", help="Metric used for checkpoint selection and early stopping.")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Minimum improvement required for checkpoint selection.")
    parser.add_argument("--lr-scheduler", choices=["none", "plateau"], default="none", help="Optional ReduceLROnPlateau scheduler.")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="Learning-rate reduction factor for plateau scheduler.")
    parser.add_argument("--lr-patience", type=int, default=3, help="Plateau scheduler patience in epochs.")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Minimum learning rate for plateau scheduler.")
    parser.add_argument("--pt-floor", type=float, default=1e-3)
    parser.add_argument("--no-pt-log-scale", dest="pt_log_scale", action="store_false")
    parser.set_defaults(pt_log_scale=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    out_dir = os.path.dirname(args.output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
