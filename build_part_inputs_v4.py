#!/usr/bin/env python3
"""
Build v3 Particle Transformer inputs with a clean model definition.

One output parquet contains both:
- core reconstructed objects: 2 leptons, 2 tagging jets, MET
- low-level constituents: EFlowTrack + EFlowPhoton + EFlowNeutralHadron

Training modes then select:
- low: constituents only
- high: core objects only
- opt: core objects + constituents, plus physics-informed edge/global features

Python 3.8 compatible.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import awkward as ak
import numpy as np
import pandas as pd
import uproot
import vector


vector.register_awkward()

DEFAULT_SAMPLES = {
    "ww": {
        "/home/public/vbs_project/MG_sample/WWjj_EW/Events/run_03/tag_1_delphes_events.root": 1,
        "/home/public/vbs_project/MG_sample/WWjj_QCD/Events/run_03/tag_1_delphes_events.root": 0,
        "/home/public/vbs_project/MG_sample/WZjj_EW/Events/run_03/tag_1_delphes_events.root": 0,
        "/home/public/vbs_project/MG_sample/WZjj_QCD/Events/run_03/tag_1_delphes_events.root": 0,
    },
    "ll": {
        "/home/public/vbs_project/MG_sample/WWjj_EW-LL-WW_cmf/Events/run_03/tag_1_delphes_events.root": 1,
        "/home/public/vbs_project/MG_sample/WWjj_EW-LT-WW_cmf/Events/run_03/tag_1_delphes_events.root": 0,
        "/home/public/vbs_project/MG_sample/WWjj_EW-TT-WW_cmf/Events/run_03/tag_1_delphes_events.root": 0,
    },
    "lt": {
        "/home/public/vbs_project/MG_sample/WWjj_EW-LL-WW_cmf/Events/run_03/tag_1_delphes_events.root": 0,
        "/home/public/vbs_project/MG_sample/WWjj_EW-LT-WW_cmf/Events/run_03/tag_1_delphes_events.root": 1,
        "/home/public/vbs_project/MG_sample/WWjj_EW-TT-WW_cmf/Events/run_03/tag_1_delphes_events.root": 0,
    },
    "lx": {
        "/home/public/vbs_project/MG_sample/WWjj_EW-LL-WW_cmf/Events/run_03/tag_1_delphes_events.root": 1,
        "/home/public/vbs_project/MG_sample/WWjj_EW-LT-WW_cmf/Events/run_03/tag_1_delphes_events.root": 1,
        "/home/public/vbs_project/MG_sample/WWjj_EW-TT-WW_cmf/Events/run_03/tag_1_delphes_events.root": 0,
    },
}

TYPE_ID = {
    "padding": 0,
    "met": 1,
    "lepton": 2,
    "jet": 3,
    "eflow_track": 4,
    "eflow_photon": 5,
    "eflow_neutral_hadron": 6,
}


def process_name_from_path(path: str) -> str:
    return path.split("/")[5].replace("-", "_")


def has_branch(tree: Any, branch: str) -> bool:
    try:
        tree[branch]
        return True
    except Exception:
        return False


def load_sr_selector(sr_module: Optional[str]):
    if not sr_module:
        return None
    module_path = os.path.abspath(sr_module)
    if os.path.isdir(module_path):
        module_path = os.path.join(module_path, "MG_signal_background.py")
    spec = importlib.util.spec_from_file_location("mg_signal_background_v3", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not import SR module: %s" % module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "select_SR"):
        raise AttributeError("%s does not define select_SR" % module_path)
    return module.select_SR


def build_reco_objects(tree: Any):
    e_pt = tree["Electron.PT"].array()
    electrons = vector.zip(
        {
            "pt": e_pt,
            "eta": tree["Electron.Eta"].array(),
            "phi": tree["Electron.Phi"].array(),
            "mass": ak.ones_like(e_pt) * 0.000511,
            "charge": tree["Electron.Charge"].array(),
            "flavor": ak.ones_like(e_pt) * 11,
        }
    )
    mu_pt = tree["Muon.PT"].array()
    muons = vector.zip(
        {
            "pt": mu_pt,
            "eta": tree["Muon.Eta"].array(),
            "phi": tree["Muon.Phi"].array(),
            "mass": ak.ones_like(mu_pt) * 0.105,
            "charge": tree["Muon.Charge"].array(),
            "flavor": ak.ones_like(mu_pt) * 13,
        }
    )
    j_pt = tree["Jet.PT"].array()
    jets = vector.zip(
        {
            "pt": j_pt,
            "eta": tree["Jet.Eta"].array(),
            "phi": tree["Jet.Phi"].array(),
            "mass": tree["Jet.Mass"].array(),
            "rapidity": tree["Jet.Eta"].array(),
            "btag": tree["Jet.BTag"].array(),
        }
    )
    return electrons, muons, jets


def pad2(array: Any, count: int, fill: float = 0.0) -> np.ndarray:
    return ak.to_numpy(ak.fill_none(ak.pad_none(array, count, axis=1, clip=True), fill))


def add_sequence(
    data: Dict[str, Any],
    prefix: str,
    pt: Any,
    eta: Any,
    phi: Any,
    mass: Any,
    type_id: Any,
    count: int,
) -> None:
    pt_p = pad2(pt, count, 0.0)
    eta_p = pad2(eta, count, 0.0)
    phi_p = pad2(phi, count, 0.0)
    mass_p = pad2(mass, count, 0.0)
    present = (pt_p > 0).astype(np.float32)
    if np.isscalar(type_id):
        type_p = np.full_like(pt_p, int(type_id), dtype=np.int64)
        type_p[present <= 0] = TYPE_ID["padding"]
    else:
        type_p = pad2(type_id, count, TYPE_ID["padding"]).astype(np.int64)
    for i in range(count):
        data["%s%d_pt" % (prefix, i)] = pt_p[:, i]
        data["%s%d_eta" % (prefix, i)] = eta_p[:, i]
        data["%s%d_phi" % (prefix, i)] = phi_p[:, i]
        data["%s%d_m" % (prefix, i)] = mass_p[:, i]
        data["%s%d_type" % (prefix, i)] = type_p[:, i]
        data["%s%d_mask" % (prefix, i)] = present[:, i]


def eflow_arrays(tree: Any, mask: Any, max_constituents: int) -> Tuple[Any, Any, Any, Any, Any, Dict[str, int]]:
    pts, etas, phis, masses, types = [], [], [], [], []
    counts: Dict[str, int] = {}
    specs = [
        ("EFlowTrack", "PT", TYPE_ID["eflow_track"]),
        ("EFlowPhoton", "ET", TYPE_ID["eflow_photon"]),
        ("EFlowNeutralHadron", "ET", TYPE_ID["eflow_neutral_hadron"]),
    ]
    for coll, pt_name, type_id in specs:
        pt_branch = "%s.%s" % (coll, pt_name)
        eta_branch = "%s.Eta" % coll
        phi_branch = "%s.Phi" % coll
        if not (has_branch(tree, pt_branch) and has_branch(tree, eta_branch) and has_branch(tree, phi_branch)):
            counts[coll] = 0
            continue
        pt = tree[pt_branch].array()[mask]
        eta = tree[eta_branch].array()[mask]
        phi = tree[phi_branch].array()[mask]
        mass = ak.zeros_like(pt)
        typ = ak.ones_like(pt) * type_id
        pts.append(pt)
        etas.append(eta)
        phis.append(phi)
        masses.append(mass)
        types.append(typ)
        counts[coll] = int(ak.sum(ak.num(pt)))
    if not pts:
        n_events = int(ak.sum(mask))
        empty = ak.Array([[] for _ in range(n_events)])
        return empty, empty, empty, empty, empty, counts
    pt_all = ak.concatenate(pts, axis=1)
    eta_all = ak.concatenate(etas, axis=1)
    phi_all = ak.concatenate(phis, axis=1)
    mass_all = ak.concatenate(masses, axis=1)
    type_all = ak.concatenate(types, axis=1)
    order = ak.argsort(pt_all, ascending=False)
    return pt_all[order][:, :max_constituents], eta_all[order][:, :max_constituents], phi_all[order][:, :max_constituents], mass_all[order][:, :max_constituents], type_all[order][:, :max_constituents], counts


def extract_file(path: str, label: int, max_constituents: int, select_sr=None):
    if not os.path.exists(path):
        return None, {"path": path, "label": label, "status": "missing"}
    process = process_name_from_path(path)
    tree = uproot.open("%s:Delphes" % path)
    n_total = len(tree["MissingET.MET"].array())
    met_pt_all = tree["MissingET.MET"].array()
    met_phi_all = tree["MissingET.Phi"].array()

    if select_sr is not None:
        electrons, muons, jets_all = build_reco_objects(tree)
        mask, leps, jets = select_sr(electrons, muons, jets_all, met_pt_all[:, 0])
        leps = leps[mask]
        jets = jets[mask]
        leps_pt, leps_eta, leps_phi, leps_m = leps.pt, leps.eta, leps.phi, leps.mass
        jets_pt, jets_eta, jets_phi, jets_m = jets.pt, jets.eta, jets.phi, jets.mass
    else:
        e_pt = tree["Electron.PT"].array()
        mu_pt = tree["Muon.PT"].array()
        leps_pt = ak.concatenate([e_pt, mu_pt], axis=1)
        leps_eta = ak.concatenate([tree["Electron.Eta"].array(), tree["Muon.Eta"].array()], axis=1)
        leps_phi = ak.concatenate([tree["Electron.Phi"].array(), tree["Muon.Phi"].array()], axis=1)
        leps_m = ak.concatenate([ak.ones_like(e_pt) * 0.000511, ak.ones_like(mu_pt) * 0.105], axis=1)
        jets_pt = tree["Jet.PT"].array()
        jets_eta = tree["Jet.Eta"].array()
        jets_phi = tree["Jet.Phi"].array()
        jets_m = tree["Jet.Mass"].array()
        mask = (ak.num(leps_pt) >= 2) & (ak.num(jets_pt) >= 2) & (ak.num(met_pt_all) >= 1)
        leps_pt, leps_eta, leps_phi, leps_m = leps_pt[mask], leps_eta[mask], leps_phi[mask], leps_m[mask]
        jets_pt, jets_eta, jets_phi, jets_m = jets_pt[mask], jets_eta[mask], jets_phi[mask], jets_m[mask]
        lep_order = ak.argsort(leps_pt, ascending=False)
        jet_order = ak.argsort(jets_pt, ascending=False)
        leps_pt, leps_eta, leps_phi, leps_m = leps_pt[lep_order], leps_eta[lep_order], leps_phi[lep_order], leps_m[lep_order]
        jets_pt, jets_eta, jets_phi, jets_m = jets_pt[jet_order], jets_eta[jet_order], jets_phi[jet_order], jets_m[jet_order]

    met_pt = met_pt_all[mask]
    met_phi = met_phi_all[mask]
    full_event_number = tree["Event.Number"].array() if has_branch(tree, "Event.Number") else ak.Array([[i] for i in range(n_total)])
    full_event_weight = tree["Event.Weight"].array() if has_branch(tree, "Event.Weight") else ak.Array([[1.0] for _ in range(n_total)])
    event_number = full_event_number[mask]
    event_weight = full_event_weight[mask]
    n_selected = len(met_pt)

    data: Dict[str, Any] = {
        "Process": process,
        "EventNumber": ak.to_numpy(event_number[:, 0]) if n_selected else np.array([], dtype=np.int64),
        "Label": label,
        "Weight": ak.to_numpy(event_weight[:, 0]) if n_selected else np.array([], dtype=np.float32),
    }

    add_sequence(data, "core", leps_pt[:, :2], leps_eta[:, :2], leps_phi[:, :2], leps_m[:, :2], TYPE_ID["lepton"], 2)
    add_sequence(data, "core", jets_pt[:, :2], jets_eta[:, :2], jets_phi[:, :2], jets_m[:, :2], TYPE_ID["jet"], 2)
    # The three calls above intentionally reused the prefix; rebuild core explicitly below.
    data_core = {k: data[k] for k in ["Process", "EventNumber", "Label", "Weight"]}
    core_pt = ak.concatenate([leps_pt[:, :2], jets_pt[:, :2], met_pt[:, :1]], axis=1)
    core_eta = ak.concatenate([leps_eta[:, :2], jets_eta[:, :2], ak.zeros_like(met_pt[:, :1])], axis=1)
    core_phi = ak.concatenate([leps_phi[:, :2], jets_phi[:, :2], met_phi[:, :1]], axis=1)
    core_m = ak.concatenate([leps_m[:, :2], jets_m[:, :2], ak.zeros_like(met_pt[:, :1])], axis=1)
    core_type = ak.concatenate(
        [
            ak.ones_like(leps_pt[:, :2]) * TYPE_ID["lepton"],
            ak.ones_like(jets_pt[:, :2]) * TYPE_ID["jet"],
            ak.ones_like(met_pt[:, :1]) * TYPE_ID["met"],
        ],
        axis=1,
    )
    add_sequence(data_core, "core", core_pt, core_eta, core_phi, core_m, core_type, 5)

    cand_pt, cand_eta, cand_phi, cand_m, cand_type, eflow_counts = eflow_arrays(tree, mask, max_constituents)
    add_sequence(data_core, "cand", cand_pt, cand_eta, cand_phi, cand_m, cand_type, max_constituents)

    meta = {
        "path": path,
        "process": process,
        "label": label,
        "events_total": int(n_total),
        "events_selected": int(n_selected),
        "selection": "select_SR" if select_sr is not None else ">=2 leptons, >=2 jets, >=1 MET",
        "max_constituents": int(max_constituents),
        "eflow_total_counts": eflow_counts,
    }
    return pd.DataFrame(data_core).fillna(0), meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(DEFAULT_SAMPLES), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-constituents", type=int, default=128)
    parser.add_argument("--apply-sr-cuts", action="store_true")
    parser.add_argument("--sr-module", default="/home/Joanne/VBS/MG_signal_background.py")
    args = parser.parse_args()

    select_sr = load_sr_selector(args.sr_module) if args.apply_sr_cuts else None
    frames, samples = [], []
    for path, label in DEFAULT_SAMPLES[args.task].items():
        print("[read] %s" % process_name_from_path(path))
        frame, meta = extract_file(path, label, args.max_constituents, select_sr=select_sr)
        samples.append(meta)
        if frame is not None:
            frames.append(frame)
    if not frames:
        raise SystemExit("No events extracted.")
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(args.output)
    manifest = {
        "task": args.task,
        "output": args.output,
        "rows": int(len(out)),
        "columns": list(out.columns),
        "label_counts": {str(k): int(v) for k, v in out["Label"].value_counts().to_dict().items()},
        "process_counts": {str(k): int(v) for k, v in out["Process"].value_counts().to_dict().items()},
        "type_id": TYPE_ID,
        "core_objects": ["leading lepton", "subleading lepton", "leading tagging jet", "subleading tagging jet", "MET"],
        "constituents": ["EFlowTrack", "EFlowPhoton", "EFlowNeutralHadron"],
        "max_constituents": int(args.max_constituents),
        "apply_sr_cuts": bool(args.apply_sr_cuts),
        "sr_module": args.sr_module if args.apply_sr_cuts else None,
        "samples": samples,
    }
    manifest_path = os.path.splitext(args.output)[0] + "_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print("[done] wrote %s" % args.output)
    print("[done] wrote %s" % manifest_path)


if __name__ == "__main__":
    main()
