#!/usr/bin/env python3
"""
Train v3 Particle Transformer models.

Definitions:
- low: constituent-level ParT. Uses EFlow constituents only, no edge features.
- high: object-level ParT. Uses 2 leptons, 2 tagging jets, MET with angular edges.
- opt: optimized physics-informed ParT. Uses core objects + EFlow constituents,
  pairwise Lorentz edge features, and global VBS features.

Python 3.8 compatible.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List

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

MODE_DEFINITIONS = {
    "low": {
        "name": "Constituent-Level ParT",
        "objects": "EFlowTrack + EFlowPhoton + EFlowNeutralHadron, up to max_constituents per event",
        "node_features": ["log10(pt_or_ET)", "eta", "phi", "log10(mass)", "type_embedding"],
        "edge_features": [],
        "global_features": [],
        "physics_meaning": "Tests whether detector-level particle-flow constituents alone can learn the event structure.",
    },
    "high": {
        "name": "Object-Level ParT",
        "objects": "leading lepton, subleading lepton, leading tagging jet, subleading tagging jet, MET",
        "node_features": ["log10(pt)", "eta", "delta_phi_to_leading_lepton", "log10(mass)", "type_embedding"],
        "edge_features": ["delta_eta", "delta_phi", "delta_R"],
        "global_features": [],
        "physics_meaning": "Uses reconstructed physics objects and their angular geometry, but no explicit invariant masses or VBS global variables.",
    },
    "opt": {
        "name": "Optimized Physics-Informed ParT",
        "objects": "core reconstructed objects + EFlow constituents",
        "node_features": ["log10(pt_or_ET)", "eta", "delta_phi_to_leading_lepton", "log10(mass)", "log10(energy)", "type_embedding"],
        "edge_features": ["delta_eta", "delta_phi", "delta_R", "log10(pair_mass)", "log10(pair_pt)", "log10(pair_transverse_mass)"],
        "global_features": ["log10(mjj)", "detajj", "zstar_lep1", "zstar_lep2", "log10(MET)"],
        "physics_meaning": "Adds pairwise Lorentz information and explicit VBS topology for best expected performance.",
    },
}


def parse_fold_list(text: str, n_folds: int) -> List[int]:
    if text.lower() == "all":
        return list(range(n_folds))
    folds: List[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, stop = [int(x) for x in item.split("-", 1)]
            folds.extend(range(start, stop + 1))
        else:
            folds.append(int(item))
    bad = [x for x in folds if x < 0 or x >= n_folds]
    if bad:
        raise ValueError("Bad folds: %s" % bad)
    return sorted(dict.fromkeys(folds))


def infer_count(df: pd.DataFrame, prefix: str) -> int:
    values = []
    for col in df.columns:
        m = re.fullmatch(r"%s(\d+)_pt" % prefix, col)
        if m:
            values.append(int(m.group(1)))
    if not values:
        raise ValueError("No %s{i}_pt columns found" % prefix)
    return max(values) + 1


def read_sequence(df: pd.DataFrame, prefix: str, count: int):
    n = len(df)
    pt = np.zeros((n, count), dtype=np.float32)
    eta = np.zeros_like(pt)
    phi = np.zeros_like(pt)
    mass = np.zeros_like(pt)
    typ = np.zeros((n, count), dtype=np.int64)
    mask = np.zeros((n, count), dtype=np.float32)
    for i in range(count):
        pt[:, i] = np.maximum(df["%s%d_pt" % (prefix, i)].to_numpy(np.float32), 0.0)
        eta[:, i] = df["%s%d_eta" % (prefix, i)].to_numpy(np.float32)
        phi[:, i] = df["%s%d_phi" % (prefix, i)].to_numpy(np.float32)
        mass[:, i] = np.maximum(df["%s%d_m" % (prefix, i)].to_numpy(np.float32), 0.0)
        typ[:, i] = df["%s%d_type" % (prefix, i)].to_numpy(np.int64)
        mask[:, i] = df["%s%d_mask" % (prefix, i)].to_numpy(np.float32)
    return pt, eta, phi, mass, typ, mask


def wrap_phi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def build_tensors(df: pd.DataFrame, mode: str, max_cands_train: int = 0):
    core_n = infer_count(df, "core")
    cand_n = infer_count(df, "cand")
    core = read_sequence(df, "core", core_n)
    cand = read_sequence(df, "cand", cand_n)
    if max_cands_train and max_cands_train > 0 and max_cands_train < cand_n:
        cand = tuple(arr[:, :max_cands_train] for arr in cand)
        cand_n = max_cands_train

    if mode == "low":
        pt, eta, phi, mass, typ, mask = cand
        phi_feature = phi
        include_energy = False
        edge_kind = "none"
        use_globals = False
    elif mode == "high":
        pt, eta, phi, mass, typ, mask = core
        phi_feature = wrap_phi(phi - phi[:, [0]])
        include_energy = False
        edge_kind = "angular"
        use_globals = False
    elif mode == "opt":
        pt = np.concatenate([core[0], cand[0]], axis=1)
        eta = np.concatenate([core[1], cand[1]], axis=1)
        phi = np.concatenate([core[2], cand[2]], axis=1)
        mass = np.concatenate([core[3], cand[3]], axis=1)
        typ = np.concatenate([core[4], cand[4]], axis=1)
        mask = np.concatenate([core[5], cand[5]], axis=1)
        phi_feature = wrap_phi(phi - core[2][:, [0]])
        include_energy = True
        edge_kind = "physics"
        use_globals = True
    else:
        raise ValueError("Unknown mode: %s" % mode)

    safe_pt = np.maximum(pt, 1e-3)
    safe_m = np.maximum(mass, 1e-3)
    energy = np.sqrt((pt * np.cosh(eta)) ** 2 + mass**2)

    node_parts = [np.log10(safe_pt), eta, phi_feature, np.log10(safe_m)]
    if include_energy:
        node_parts.append(np.log10(np.maximum(energy, 1e-3)))
    nodes = np.stack(node_parts, axis=-1).astype(np.float32)
    nodes *= mask[:, :, None]

    n_obj = pt.shape[1]
    if edge_kind == "none":
        edges = np.zeros((len(df), n_obj, n_obj, 1), dtype=np.float32)
    else:
        d_eta = eta[:, :, None] - eta[:, None, :]
        d_phi = wrap_phi(phi[:, :, None] - phi[:, None, :])
        d_r = np.sqrt(d_eta**2 + d_phi**2)
        if edge_kind == "angular":
            edges = np.stack([d_eta, d_phi, d_r], axis=-1).astype(np.float32)
        else:
            px = pt * np.cos(phi)
            py = pt * np.sin(phi)
            pz = pt * np.sinh(eta)
            et = np.sqrt(pt**2 + mass**2)
            e_ij = energy[:, :, None] + energy[:, None, :]
            px_ij = px[:, :, None] + px[:, None, :]
            py_ij = py[:, :, None] + py[:, None, :]
            pz_ij = pz[:, :, None] + pz[:, None, :]
            pt_ij = np.sqrt(px_ij**2 + py_ij**2)
            m2_ij = e_ij**2 - px_ij**2 - py_ij**2 - pz_ij**2
            mt2_ij = (et[:, :, None] + et[:, None, :]) ** 2 - pt_ij**2
            edges = np.stack(
                [
                    d_eta,
                    d_phi,
                    d_r,
                    np.log10(np.sqrt(np.maximum(m2_ij, 1e-3))),
                    np.log10(np.maximum(pt_ij, 1e-3)),
                    np.log10(np.sqrt(np.maximum(mt2_ij, 1e-3))),
                ],
                axis=-1,
            ).astype(np.float32)
    edges *= mask[:, :, None, None] * mask[:, None, :, None]

    globals_ = np.zeros((len(df), 5), dtype=np.float32)
    if use_globals:
        core_pt, core_eta, core_phi, core_m = core[0], core[1], core[2], core[3]
        j1, j2 = 2, 3
        e_core = np.sqrt((core_pt * np.cosh(core_eta)) ** 2 + core_m**2)
        px = core_pt * np.cos(core_phi)
        py = core_pt * np.sin(core_phi)
        pz = core_pt * np.sinh(core_eta)
        jj_e = e_core[:, j1] + e_core[:, j2]
        jj_px = px[:, j1] + px[:, j2]
        jj_py = py[:, j1] + py[:, j2]
        jj_pz = pz[:, j1] + pz[:, j2]
        mjj = np.sqrt(np.maximum(jj_e**2 - jj_px**2 - jj_py**2 - jj_pz**2, 1e-3))
        detajj = np.abs(core_eta[:, j1] - core_eta[:, j2])
        center = 0.5 * (core_eta[:, j1] + core_eta[:, j2])
        globals_ = np.column_stack(
            [
                np.log10(mjj),
                detajj,
                np.abs(core_eta[:, 0] - center) / np.maximum(detajj, 1e-3),
                np.abs(core_eta[:, 1] - center) / np.maximum(detajj, 1e-3),
                np.log10(np.maximum(core_pt[:, 4], 1e-3)),
            ]
        ).astype(np.float32)

    feature_manifest = dict(MODE_DEFINITIONS[mode])
    feature_manifest.update(
        {
            "node_shape_per_event": [int(nodes.shape[1]), int(nodes.shape[2])],
            "edge_shape_per_event": [int(edges.shape[1]), int(edges.shape[2]), int(edges.shape[3])],
            "global_shape_per_event": [int(globals_.shape[1])],
            "core_object_count": int(core_n),
            "constituent_count": int(cand_n),
            "max_cands_train": int(max_cands_train) if max_cands_train else "all",
            "uses_globals": bool(use_globals),
        }
    )
    return nodes, typ, edges, globals_, mask.astype(np.float32), feature_manifest


def apply_training_weights(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Weight"] = out["Weight"] * out["Process"].map(PROCESS_WEIGHTS).fillna(1.0)
    out["Train_Weight"] = np.abs(out["Weight"])
    sig = out.loc[out["Label"] == 1, "Train_Weight"].sum()
    bkg = out.loc[out["Label"] == 0, "Train_Weight"].sum()
    if sig > 0 and bkg > 0:
        out.loc[out["Label"] == 1, "Train_Weight"] *= bkg / sig
    return out


class ParTDataset(Dataset):
    def __init__(self, x, c, v, g, mask, y, w):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.c = torch.tensor(c, dtype=torch.long)
        self.v = torch.tensor(v, dtype=torch.float32)
        self.g = torch.tensor(g, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        self.w = torch.tensor(w, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.c[i], self.v[i], self.g[i], self.mask[i], self.y[i], self.w[i]


class AttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def forward(self, x, bias, key_mask):
        b, n, dim = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim) + bias
        attn = attn.masked_fill(~key_mask[:, None, None, :], -1e4)
        attn = torch.softmax(attn, dim=-1)
        h = (self.drop(attn) @ v).transpose(1, 2).reshape(b, n, dim)
        x = x + self.drop(self.proj(h))
        return x + self.mlp(self.norm2(x))


class ParticleTransformer(nn.Module):
    def __init__(self, node_dim, edge_dim, n_types, use_globals, dim=64, heads=4, layers=4, dropout=0.2):
        super().__init__()
        self.use_globals = use_globals
        self.node = nn.Linear(node_dim, dim - 16)
        self.kind = nn.Embedding(max(int(n_types), 8) + 1, 16)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.edge = nn.Sequential(nn.Linear(edge_dim, 32), nn.GELU(), nn.Linear(32, heads))
        self.blocks = nn.ModuleList([AttentionBlock(dim, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        head_in = dim + (5 if use_globals else 0)
        self.head = nn.Sequential(nn.Linear(head_in, dim // 2), nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim // 2, 1), nn.Sigmoid())

    def forward(self, x, c, v, g, mask):
        b, n, _ = x.shape
        h = torch.cat([self.node(x), self.kind(c)], dim=-1)
        bias = self.edge(v).permute(0, 3, 1, 2)
        bias_pad = torch.zeros(b, bias.shape[1], n + 1, n + 1, device=x.device)
        bias_pad[:, :, 1:, 1:] = bias
        full_mask = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=x.device), mask], dim=1)
        h = torch.cat([self.cls.expand(b, -1, -1), h], dim=1)
        for block in self.blocks:
            h = block(h, bias_pad, full_mask)
        pooled = self.norm(h[:, 0])
        if self.use_globals:
            pooled = torch.cat([pooled, g], dim=-1)
        return self.head(pooled)


@dataclass
class Split:
    x: np.ndarray
    c: np.ndarray
    v: np.ndarray
    g: np.ndarray
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
    losses, y_true, y_pred, weights = [], [], [], []
    with torch.no_grad():
        for xb, cb, vb, gb, mb, yb, wb in loader:
            xb, cb, vb, gb, mb = xb.to(device), cb.to(device), vb.to(device), gb.to(device), mb.to(device)
            yb, wb = yb.to(device), wb.to(device)
            pred = model(xb, cb, vb, gb, mb)
            losses.append(float((crit(pred, yb) * (wb / wb.mean())).mean().cpu()))
            y_true.extend(yb.cpu().numpy().ravel())
            y_pred.extend(pred.cpu().numpy().ravel())
            weights.extend(wb.cpu().numpy().ravel())
    return float(np.mean(losses)), safe_auc(y_true, y_pred, weights)


def load_state_dict_compat(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def plot_history(history: Dict[str, Any], out_prefix: str, mode: str):
    folds = history.get("folds", {})
    if not folds:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for fold, vals in sorted(folds.items(), key=lambda x: int(x[0])):
        axes[0].plot(vals["epoch"], vals["train_loss"], label="fold %s train" % fold)
        axes[0].plot(vals["epoch"], vals["val_loss"], linestyle="--", label="fold %s val" % fold)
        axes[1].plot(vals["epoch"], vals["train_auc"], label="fold %s train" % fold)
        axes[1].plot(vals["epoch"], vals["val_auc"], linestyle="--", label="fold %s val" % fold)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Weighted BCE loss")
    axes[0].set_title("%s loss" % mode)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Weighted ROC AUC")
    axes[1].set_title("%s AUC" % mode)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("%s_%s_training_curves.png" % (out_prefix, mode), dpi=200)
    fig.savefig("%s_%s_training_curves.pdf" % (out_prefix, mode))
    plt.close(fig)


def train(args):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    df = apply_training_weights(pd.read_parquet(args.input))
    if "Fold_ID" not in df.columns:
        df["Fold_ID"] = np.arange(len(df)) % args.folds
    x, c, v, g, mask, feature_manifest = build_tensors(df, args.mode, args.max_cands_train)
    y = df["Label"].to_numpy(np.float32)
    w = df["Train_Weight"].to_numpy(np.float32)

    scored_path = "%s_%s_scored.parquet" % (args.output_prefix, args.mode)
    hist_path = "%s_%s_history.json" % (args.output_prefix, args.mode)
    df[args.score_column] = np.nan
    if args.resume and os.path.exists(scored_path):
        old = pd.read_parquet(scored_path)
        if len(old) == len(df) and args.score_column in old.columns:
            df[args.score_column] = old[args.score_column].to_numpy()
    if args.resume and os.path.exists(hist_path):
        with open(hist_path, "r", encoding="utf-8") as handle:
            history = json.load(handle)
    else:
        history = {
            "folds": {},
            "fold_auc": [],
            "feature_manifest": feature_manifest,
            "mode_definition": MODE_DEFINITIONS[args.mode],
            "run_config": vars(args),
            "weighting": {
                "process_weights": PROCESS_WEIGHTS,
                "method": "Weight *= process factor; Train_Weight=abs(Weight); signal Train_Weight rescaled to match background sum.",
                "train_weight_label_sums": {str(k): float(vv) for k, vv in df.groupby("Label")["Train_Weight"].sum().to_dict().items()},
                "weighted_process_sums": {str(k): float(vv) for k, vv in df.groupby("Process")["Weight"].sum().to_dict().items()},
            },
        }

    fold_values = parse_fold_list(args.fold_list, args.folds)
    for fold in fold_values:
        if args.resume and str(fold) in history.get("folds", {}) and not args.overwrite_fold:
            print("fold %d already complete, skipping" % fold)
            continue
        test = df["Fold_ID"].to_numpy() == fold
        val = df["Fold_ID"].to_numpy() == ((fold + 1) % args.folds)
        train_mask = ~(test | val)

        sx = StandardScaler().fit(x[train_mask].reshape(-1, x.shape[-1]))
        sv = StandardScaler().fit(v[train_mask].reshape(-1, v.shape[-1]))
        sg = StandardScaler().fit(g[train_mask])

        def make(sel):
            b = int(sel.sum())
            return Split(
                sx.transform(x[sel].reshape(-1, x.shape[-1])).reshape(b, x.shape[1], x.shape[2]),
                c[sel],
                sv.transform(v[sel].reshape(-1, v.shape[-1])).reshape(b, v.shape[1], v.shape[2], v.shape[3]),
                sg.transform(g[sel]),
                mask[sel],
                y[sel],
                w[sel],
            )

        tr, va, te = make(train_mask), make(val), make(test)
        train_loader = DataLoader(ParTDataset(*tr.__dict__.values()), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(ParTDataset(*va.__dict__.values()), batch_size=args.batch_size)
        test_loader = DataLoader(ParTDataset(*te.__dict__.values()), batch_size=args.batch_size)

        model = ParticleTransformer(x.shape[-1], v.shape[-1], int(c.max()), args.mode == "opt", args.dim, args.heads, args.layers, args.dropout).to(device)
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        crit = nn.BCELoss(reduction="none")
        best_loss = float("inf")
        patience = 0
        best_path = "%s_%s_fold%d.pt" % (args.output_prefix, args.mode, fold)
        prep_path = "%s_%s_fold%d_preprocess.pkl" % (args.output_prefix, args.mode, fold)
        with open(prep_path, "wb") as handle:
            pickle.dump(
                {
                    "scaler_x": sx,
                    "scaler_v": sv,
                    "scaler_g": sg,
                    "mode": args.mode,
                    "node_dim": x.shape[-1],
                    "edge_dim": v.shape[-1],
                    "n_types": int(c.max()),
                    "dim": args.dim,
                    "heads": args.heads,
                    "layers": args.layers,
                    "dropout": args.dropout,
                    "max_cands_train": args.max_cands_train,
                },
                handle,
            )

        fold_hist = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_auc": [],
            "val_auc": [],
            "best_epoch": None,
            "split_counts": {"train": int(train_mask.sum()), "validation": int(val.sum()), "test": int(test.sum())},
            "split_weight_sums": {"train": float(w[train_mask].sum()), "validation": float(w[val].sum()), "test": float(w[test].sum())},
        }

        for epoch in range(args.epochs):
            model.train()
            for xb, cb, vb, gb, mb, yb, wb in train_loader:
                xb, cb, vb, gb, mb = xb.to(device), cb.to(device), vb.to(device), gb.to(device), mb.to(device)
                yb, wb = yb.to(device), wb.to(device)
                opt.zero_grad()
                pred = model(xb, cb, vb, gb, mb)
                loss = (crit(pred, yb) * (wb / wb.mean())).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            train_loss, train_auc = evaluate(model, train_loader, crit, device)
            val_loss, val_auc = evaluate(model, val_loader, crit, device)
            fold_hist["epoch"].append(epoch + 1)
            fold_hist["train_loss"].append(train_loss)
            fold_hist["val_loss"].append(val_loss)
            fold_hist["train_auc"].append(train_auc)
            fold_hist["val_auc"].append(val_auc)
            print("fold %d epoch %03d train_loss=%.4f val_loss=%.4f train_auc=%.4f val_auc=%.4f" % (fold, epoch + 1, train_loss, val_loss, train_auc, val_auc))
            if val_loss < best_loss:
                best_loss = val_loss
                patience = 0
                fold_hist["best_epoch"] = epoch + 1
                torch.save(model.state_dict(), best_path)
            else:
                patience += 1
            if patience >= args.patience:
                break

        model.load_state_dict(load_state_dict_compat(best_path, device))
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, cb, vb, gb, mb, _, _ in test_loader:
                preds.extend(model(xb.to(device), cb.to(device), vb.to(device), gb.to(device), mb.to(device)).cpu().numpy().ravel())
        df.loc[test, args.score_column] = np.asarray(preds)
        test_auc = safe_auc(y[test], preds, w[test])
        fold_hist["test_auc"] = test_auc
        history["folds"][str(fold)] = fold_hist
        history["fold_auc"] = [vals["test_auc"] for _, vals in sorted(history["folds"].items(), key=lambda item: int(item[0]))]
        history["mean_auc"] = float(np.mean(history["fold_auc"]))
        history["std_auc"] = float(np.std(history["fold_auc"]))
        history["elapsed_seconds"] = float(time.time() - start_time)
        df.to_parquet(scored_path)
        with open(hist_path, "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        plot_history(history, args.output_prefix, args.mode)
        print("fold %d test_auc=%.4f" % (fold, test_auc))
        print("updated %s and %s" % (scored_path, hist_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--mode", choices=["low", "high", "opt"], required=True)
    parser.add_argument("--score-column", default="part_score")
    parser.add_argument("--fold-list", default="all")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite-fold", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-cands-train", type=int, default=0, help="Use only the leading N EFlow constituents during training/scoring. 0 means all stored constituents.")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    out_dir = os.path.dirname(args.output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
