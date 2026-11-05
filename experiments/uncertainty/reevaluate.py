"""
Re-evaluate old copula runs (the .pt files written by the old run.py) with the new code,
and check that the new code reproduces the old numbers.

    python -m uncertainty.reevaluate --old "<path>/results/copula (2)/copula"
    python -m uncertainty.reevaluate --old "<path>" --only spiral_copula_mix=0.75

For every old run it rebuilds the world, the observer and the encoder, runs evaluate(),
and saves the result in the new format (same files as uncertainty.run writes) to --out.
It also compares new against old:

  observer_err     max |x_new - x_old| on the saved plotting points       (must be ~0)
  h_err            max |h_new - h_old| on the probe states                 (must be ~0)
  geometry_err     median |log det J new - old| on the probe states        (must be ~0)
  residual_match   rank correlation of the new and the old residual map    (random)
  residual_floor   the same between two new residual maps with different
                   sampling seeds: how close two random maps can get       (random)
  d_rho_*          new rho minus old rho, for the readouts the old run saved

"Same up to random states" means: the first three are ~0, the geometry rhos agree
exactly, and residual_match is about as high as residual_floor.
A table of all comparisons is written to <out>/comparison.csv.
"""

import argparse
import csv
import glob
import os

import numpy as np
import torch

from lejepa_id.models import make_matched_encoder, make_mlp_encoder

from . import estimators as est
from .evaluate import evaluate, spearman
from .observers import observer_from_saved_run
from .run import save
from .worlds import CopulaWorld


def rebuild_encoder(p):
    if p["encoder"] == "matched":
        encoder = make_matched_encoder(2, n_layers=p["n_layers"], device="cpu")
    else:                      # the old runner always built a 4-layer MLP
        encoder = make_mlp_encoder(2, hidden=p.get("hidden") or 256, device="cpu")
    encoder.load_state_dict(p["model_state_dict"])
    return encoder.eval()


def residual_map(encoder, observer, world, z, seed, n_views=256):
    torch.manual_seed(seed)
    views = world.sample_next(z, n_views)
    h = est.embed_states(encoder, observer, views.reshape(-1, 2)).reshape(n_views, len(z), -1)
    return est.residual(est.residual_covariance(h))


def reevaluate(path, out_dir):
    p = torch.load(path, map_location="cpu", weights_only=False)
    name = os.path.basename(path)[:-3]
    world = CopulaWorld(p.get("rho") or 0.95, p["mix"])
    observer = observer_from_saved_run(p)
    encoder = rebuild_encoder(p)

    maps, scores = evaluate(encoder, observer, world)

    # --- compare with the old run ---------------------------------------------------
    z_old = torch.as_tensor(p["z"][:2000]).float()
    cmp = {"name": name,
           "observer_err": (observer.observe(z_old) - torch.as_tensor(p["x"][:2000])).abs().max().item()}
    assert torch.allclose(maps["z"], p["field_z_probe"].float()), "probe states differ"
    cmp["h_err"] = (maps["h"] - p["field_h_probe"]).abs().max().item()
    old_ldj = p["field_log_det_J"].double()
    new_ldj = torch.linalg.slogdet(maps["J"].double())[1]
    cmp["geometry_err"] = (new_ldj - old_ldj).abs().median().item()

    old_residual = 2 * p["field_a_hat"].double().log()
    second = residual_map(encoder, observer, world, maps["z"], seed=1)
    cmp["residual_match"] = spearman(maps["residual"], old_residual)
    cmp["residual_floor"] = spearman(maps["residual"], second)
    truth = maps["true_noise"]
    old = {"residual": old_residual, "geometry": -2 * old_ldj,
           "residual + geometry": old_residual - 2 * old_ldj}
    for key, old_map in old.items():
        cmp[f"d_rho_{key}"] = scores[f"rho: {key}"] - spearman(old_map, truth)

    # --- save in the new format ------------------------------------------------------
    keep = ["mixing", "encoder", "mode", "lamb", "sigma", "hidden", "n_layers", "seed", "steps",
            "r2_zh", "r2_hz", "orth_err", "procrustes_mse", "r2_hz_grid",
            "final_align", "final_sigreg", "final_whiten", "final_loss"]
    result = {"name": name, "run": name.split("_copula")[0], "observer": p["mixing"],
              **{k: p.get(k) for k in keep}, "world": {"type": "copula", "rho": world.rho, "mix": world.mix},
              "world_tag": world.tag, "contrast": world.contrast(), "rho": world.rho,
              "reevaluated_from": path, **scores}
    plot = (torch.as_tensor(p["z"]), torch.as_tensor(p["x"]), torch.as_tensor(p["h"]))
    save(out_dir, name, result, p.get("log"), encoder, maps, plot)
    return cmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="folder with the old .pt files")
    ap.add_argument("--out", default="results/uncertainty/reevaluated")
    ap.add_argument("--only", nargs="*", default=None, help="substrings of run names to include")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.old, "*.pt")))
    if a.only:
        files = [f for f in files if any(s in os.path.basename(f) for s in a.only)]
    rows = []
    for i, f in enumerate(files):
        c = reevaluate(f, a.out)
        rows.append(c)
        print(f"[{i + 1}/{len(files)}] {c['name']:<40s} h {c['h_err']:.1e}  geometry {c['geometry_err']:.1e}  "
              f"residual match {c['residual_match']:.3f} (floor {c['residual_floor']:.3f})", flush=True)

    with open(os.path.join(a.out, "comparison.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def col(k):
        return np.array([r[k] for r in rows], dtype=float)
    print(f"\n{len(rows)} runs")
    print(f"  observer_err  max {np.nanmax(col('observer_err')):.1e}")
    print(f"  h_err         max {np.nanmax(col('h_err')):.1e}")
    print(f"  geometry_err  max {np.nanmax(col('geometry_err')):.1e}")
    for k in [k for k in rows[0] if k.startswith("d_rho_")]:
        print(f"  {k:<32s} mean |d| {np.nanmean(np.abs(col(k))):.4f}   max |d| {np.nanmax(np.abs(col(k))):.4f}")
    gap = col("residual_floor") - col("residual_match")
    print(f"  residual: floor - match  mean {np.nanmean(gap):+.4f}   max {np.nanmax(gap):+.4f}")
    print("wrote", os.path.abspath(os.path.join(a.out, "comparison.csv")))


if __name__ == "__main__":
    main()
