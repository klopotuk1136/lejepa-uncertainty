"""
Recompute the Jacobian readouts of saved Reacher runs with another finite-difference step,
without retraining.

    python -m uncertainty.redo_jacobian --config uncertainty/configs/reacher.yaml
    python -m uncertainty.redo_jacobian --config uncertainty/configs/reacher.yaml --fd_eps 0.2 0.5

On Reacher, J = dh/dz is a finite difference through the renderer (estimators.jacobian),
and the result depends on its step fd_eps. Only J and what is built from it depend on the
step: the geometry readout, residual + geometry, exact recovery and the linearization
error. So for every run in the config's `out` folder this:

  1. loads the trained encoder and the saved maps,
  2. re-scores the saved maps as they are; this must give the saved scores ("check": up to
     ~1e-4 in rho, because the maps are saved in float32 and rounding breaks rank ties),
  3. recomputes J with the new step (the residual, step size and density excess maps are kept),
  4. re-scores and saves the run to <out>_fd_eps=<step>. The original files are not touched.

"J moved" is the median relative change of J against the saved one. With --fd_eps set to
the step the runs were made with (0.1 for the first server runs) it should be ~0 on the
machine that made them: a check that it renders the same frames.
"""

import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import yaml

from lejepa_id.models import make_cnn_encoder

from .evaluate import DEFAULTS, READOUTS, jacobian_readouts, score
from .observers import make_observer
from .run import NormalizedEncoder, save
from .worlds import make_world

# readouts made of one estimator map, e.g. "step size" -> "step_size"
SINGLE = {name: needed[0] for name, needed in READOUTS.items() if len(needed) == 1}
JACOBIAN_PARTS = ("geometry", "smoothed_geometry", "encoder_geometry")


def load_reacher_run(path, device):
    """The saved run (result, maps, plotting points) and its trained encoder."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["result"]["observer"] != "reacher":
        raise ValueError(f"{path} is not a Reacher run (2-D runs use autograd: no step to change)")
    state = saved["encoder_state"]
    cnn = make_cnn_encoder(d_latent=2, device="cpu")
    encoder = NormalizedEncoder(cnn, state["mean"].flatten(), state["std"].flatten())
    encoder.load_state_dict(state)
    return saved, encoder.to(device).eval()


def redo(path, observer, settings, out_dir, device):
    saved, encoder = load_reacher_run(path, device)
    result, maps = saved["result"], saved["maps"]
    world = make_world(result["world"])

    # 2. the estimator maps, read back from the saved single readouts; re-scoring them as
    #    they are must reproduce the saved scores
    parts = {part: maps[name] for name, part in SINGLE.items() if name in maps}
    rescored = score(dict(maps), parts, world, settings)
    check = max(abs(v - result[k]) for k, v in rescored.items() if np.isfinite(v))

    # 3. new J; drop every old readout map, keep z, true_noise, h, sigma_hat
    old_J = maps["J"]
    maps = {k: v for k, v in maps.items() if k not in READOUTS}
    parts = {k: v for k, v in parts.items() if k not in JACOBIAN_PARTS}
    maps["J"], geometry_parts = jacobian_readouts(encoder, observer, world, maps["z"], settings)
    J_moved = ((maps["J"] - old_J).flatten(1).norm(dim=1) / old_J.flatten(1).norm(dim=1)).median().item()

    # 4. re-score and save
    scores = score(maps, {**parts, **geometry_parts}, world, settings)
    new_result = {k: v for k, v in result.items() if not k.startswith(("rho: ", "R2: "))}
    new_result.update(scores, evaluation=settings, jacobian_redone_from=path)
    plot = saved["plot"]
    save(out_dir, result["name"], new_result, saved["log"], encoder, maps, (plot["z"], plot["x"], plot["h"]))
    return {"name": result["name"], "contrast": result["contrast"], "check": check, "J_moved": J_moved,
            "old": result, "new": new_result}


SHOW = [("rho: geometry", "rho geometry"), ("rho: residual + geometry", "rho residual + geometry"),
        ("R2: residual + geometry", "R2 residual + geometry"), ("linearization_error", "linearization error")]


def print_row(r):
    changes = "   ".join(f"{label} {r['old'][k]:+.2f} -> {r['new'][k]:+.2f}" for k, label in SHOW)
    print(f"{r['name']:<34s} check {r['check']:.0e}  J moved {r['J_moved']:5.0%}   {changes}", flush=True)


def print_seed_means(rows):
    """Seed means per noise contrast, old step -> new step."""
    by_contrast = defaultdict(list)
    for r in rows:
        by_contrast[round(r["contrast"], 1)].append(r)
    print(f"\n{'contrast':>8s} {'seeds':>5s}  " + "  ".join(f"{label:>26s}" for _, label in SHOW))
    for contrast in sorted(by_contrast):
        rs = by_contrast[contrast]
        cells = []
        for k, _ in SHOW:
            old, new = np.mean([r["old"][k] for r in rs]), np.mean([r["new"][k] for r in rs])
            cells.append(f"{old:+.2f} -> {new:+.2f}")
        print(f"{contrast:7.1f}x {len(rs):5d}  " + "  ".join(f"{c:>26s}" for c in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="the config the runs were made with")
    ap.add_argument("--fd_eps", type=float, nargs="+", help="steps to use (default: the config's fd_eps)")
    ap.add_argument("--only", nargs="*", help="substrings of run names to include")
    a = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings = {**DEFAULTS, **(cfg.get("evaluation") or {})}
    steps = a.fd_eps or [settings["fd_eps"]]
    if None in steps:
        raise SystemExit("set fd_eps in the config's evaluation settings or pass --fd_eps")

    files = sorted(glob.glob(os.path.join(cfg["out"], "*.pt")))
    if a.only:
        files = [f for f in files if any(s in os.path.basename(f) for s in a.only)]
    observer = make_observer({"observer": "reacher"})       # one renderer for every run

    for eps in steps:
        out_dir = f"{cfg['out']}_fd_eps={eps}"
        print(f"\n=== fd_eps {eps}: {len(files)} runs from {cfg['out']} -> {out_dir}")
        rows = []
        for path in files:
            rows.append(redo(path, observer, {**settings, "fd_eps": eps}, out_dir, device))
            print_row(rows[-1])
        print_seed_means(rows)


if __name__ == "__main__":
    main()
