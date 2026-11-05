"""
Pictures from saved results (uncertainty.run or uncertainty.reevaluate output). Nothing is
recomputed except the true noise used to colour the latent pictures.

    python -m uncertainty.plots world    results/uncertainty/copula/spiral_copula_mix=0.75_seed=1337.pt
    python -m uncertainty.plots readouts results/uncertainty/copula/spiral_copula_mix=0.75_seed=1337.pt
    python -m uncertainty.plots sweep    results/uncertainty/copula

world     true latent z | what the encoder sees x | what it learned h, coloured by the true
          noise scale sqrt(tr Sigma*(z)). Reacher: example frames instead of x.
          -> <out>/Latent structure/<name>.png
readouts  true noise and readout maps over the probe states, each panel coloured by the
          rank of its own values. Titles give rho, averaged over every seed of the same setup
          found next to the file.  -> <out>/Uncertainty/<name>.png
sweep     rho (and identifiability) against noise contrast, one line per run, for every
          .json in the folder.  -> <out>/sweep/by_world.png and by_readout.png
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .worlds import make_world

# names shown in the figures
LABELS = {"residual": "residual alone",
          "geometry": "Jacobian alone",
          "residual + geometry": "residual + Jacobian",
          "smoothed geometry": "smoothed Jacobian alone",
          "residual + smoothed geometry": "residual + smoothed Jacobian",
          "encoder-only geometry": "encoder-only Jacobian",
          "step size": "step size",
          "density excess": "density excess",
          "step size + density excess": "step size + density excess"}
DEFAULT_PANELS = ["residual", "residual + geometry", "residual + smoothed geometry"]


def rank01(x):
    """Ranks scaled to (0, 1): colours compare positions within a panel, not values."""
    x = np.asarray(x)
    return (np.argsort(np.argsort(x)) + 0.5) / len(x)


def colorbar(fig, sc, label, ticks=None, ticklabels=None):
    cax = fig.add_axes([0.915, 0.18, 0.013, 0.64])
    cb = fig.colorbar(sc, cax=cax)
    if ticks is not None:
        cb.set_ticks(ticks, labels=ticklabels)
    cb.set_label(label, fontsize=9)


def title_of(r):
    return f"{r['run']}, noise contrast {r['contrast']:.1f}×, seed {r['seed']}"


# =============================================================================
# world: the latent structure
# =============================================================================

def world_figure(saved):
    r, plot = saved["result"], saved["plot"]
    z, x, h = plot["z"], plot["x"], plot["h"]
    sigma = make_world(r["world"]).true_noise(z.double()).diagonal(dim1=-2, dim2=-1).sum(-1).sqrt().numpy()
    order = sigma.argsort()                     # noisiest points drawn last, on top
    vmin, vmax = np.quantile(sigma, [0.005, 0.995])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    panels = [(axes[0], z, "true latent $z$"), (axes[2], h, "what the encoder learned, $h$")]
    if x is not None:
        panels.append((axes[1], x, "what the encoder sees, $x$"))
    else:
        reacher_frames(axes[1])
    for ax, data, name in panels:
        data = np.asarray(data)
        sc = ax.scatter(data[order, 0], data[order, 1], c=sigma[order], s=2, cmap="inferno",
                        vmin=vmin, vmax=vmax, linewidths=0)
        ax.set_title(name, fontsize=10)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
    fig.suptitle(title_of(r), fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.90, 0.96])
    colorbar(fig, sc, "how unpredictable the state is")
    return fig


def reacher_frames(ax, grid=(-2.0, 0.0, 2.0)):
    """A 3 x 3 montage of rendered frames at joint angles from `grid`."""
    from .observers import ReacherCamera
    try:
        camera = ReacherCamera()
    except Exception as e:  # noqa: BLE001 - the montage is optional
        ax.text(0.5, 0.5, f"no renderer:\n{e}", ha="center", va="center", fontsize=8)
        ax.axis("off")
        return
    rows = []
    for b in grid[::-1]:
        frames = camera.observe(torch.tensor([[a, b] for a in grid]))
        rows.append(np.concatenate([f.permute(1, 2, 0).numpy() for f in frames], axis=1))
    ax.imshow(np.concatenate(rows, axis=0))
    ax.set_xticks([32, 96, 160], labels=[f"{g:g}" for g in grid])
    ax.set_yticks([32, 96, 160], labels=[f"{g:g}" for g in grid[::-1]])
    ax.set_xlabel("shoulder angle $z_0$")
    ax.set_ylabel("wrist angle $z_1$")
    ax.set_title("what the encoder sees: 64×64 frames", fontsize=10)


# =============================================================================
# readouts: the uncertainty maps
# =============================================================================

def seed_mean_rho(path, key):
    """rho of `key`, averaged over every seed of this setup saved in the same folder."""
    base = os.path.basename(path).rsplit("_seed=", 1)[0]
    vals = []
    for f in glob.glob(os.path.join(os.path.dirname(path), base + "_seed=*.json")):
        with open(f) as fh:
            v = json.load(fh).get(f"rho: {key}")
        if v is not None and np.isfinite(v):
            vals.append(v)
    return (np.mean(vals), len(vals)) if vals else (np.nan, 0)


def readouts_figure(saved, path, panels):
    r, maps = saved["result"], saved["maps"]
    z = maps["z"].numpy()
    shown = [("true noise", maps["true_noise"].numpy(), None)]
    n_seeds = 0
    for key in [k for k in panels if k in maps]:          # e.g. Reacher has no smoothed Jacobian
        rho, n_seeds = seed_mean_rho(path, key)
        shown.append((LABELS.get(key, key), maps[key].numpy(), rho))

    step = np.min(np.diff(np.unique(z[:, 0])))           # probe grid spacing
    fig, axes = plt.subplots(1, len(shown), figsize=(4.4 * len(shown), 4.7))
    for ax, (name, values, rho) in zip(axes, shown):
        sc = ax.scatter(z[:, 0], z[:, 1], c=rank01(values), s=(step * 40) ** 2, marker="s",
                        cmap="inferno", vmin=0, vmax=1, linewidths=0)
        ax.set_title(name if rho is None else f"{name}   (ρ = {rho:+.2f})", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.set_xlim(-3.3, 3.3), ax.set_ylim(-3.3, 3.3)
    fig.suptitle(f"{title_of(r)}; every panel shows the probe states in the true latent z\n"
                 f"ρ = rank correlation with the true noise, mean of {n_seeds} seeds", fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.90, 0.93])
    colorbar(fig, sc, "rank within each panel", [0, 1], ["quietest", "noisiest"])
    return fig


# =============================================================================
# sweep: scores against noise contrast
# =============================================================================

def load_sweep(folder):
    """{run: {contrast: [result, ...]}} from every .json in the folder."""
    groups = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(os.path.join(folder, "*.json")):
        with open(f) as fh:
            r = json.load(fh)
        if "contrast" in r:
            groups[r["run"]][round(r["contrast"], 3)].append(r)
    return groups


def curve(ax, by_contrast, key, **kw):
    """Mean over seeds as a line, the seed range as a band. Skips contrasts without values."""
    xs, mean, lo, hi = [], [], [], []
    for c in sorted(by_contrast):
        v = [r[key] for r in by_contrast[c] if r.get(key) is not None and np.isfinite(r[key])]
        if v:
            xs.append(c), mean.append(np.mean(v)), lo.append(min(v)), hi.append(max(v))
    if xs:
        line, = ax.plot(xs, mean, marker="o", **kw)
        ax.fill_between(xs, lo, hi, color=line.get_color(), alpha=0.15, lw=0)
    return xs


def contrast_axis(ax, xs):
    ax.set_xscale("log")
    if xs:
        ax.set_xticks(xs, labels=[f"{x:.1f}" if x < 10 else f"{x:.0f}" for x in xs])
    ax.minorticks_off()
    ax.set_xlabel("noise contrast (noisiest ÷ quietest)")
    ax.grid(alpha=0.25)


def sweep_by_world(groups, runs, readouts):
    keys = [("r2_hz", "identifiability: R² of z from h")] + [(f"rho: {k}", f"ρ {LABELS.get(k, k)}")
                                                           for k in readouts]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 4.6))
    for ax, (key, title) in zip(axes, keys):
        xs = []
        for run in runs:
            xs = sorted(set(xs) | set(curve(ax, groups[run], key, label=run)))
        contrast_axis(ax, xs)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=8)
    fig.suptitle("mean over seeds; bands = seed range", fontsize=11)
    fig.tight_layout()
    return fig


def sweep_by_readout(groups, runs, readouts):
    fig, axes = plt.subplots(1, len(runs), figsize=(4.6 * len(runs), 4.4), sharey=True, squeeze=False)
    for ax, run in zip(axes[0], runs):
        xs = []
        for k in readouts:
            xs = sorted(set(xs) | set(curve(ax, groups[run], f"rho: {k}", label=LABELS.get(k, k))))
        contrast_axis(ax, xs)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(run, fontsize=10)
    axes[0][0].set_ylabel("ρ with the true noise")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    return fig


# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["world", "readouts", "sweep"])
    ap.add_argument("paths", nargs="+", help=".pt files (world, readouts) or one results folder (sweep)")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--panels", nargs="+", default=DEFAULT_PANELS, help="readouts to show")
    ap.add_argument("--runs", nargs="+", help="sweep: runs to draw (default: all)")
    a = ap.parse_args()

    if a.kind == "sweep":
        groups = load_sweep(a.paths[0])
        runs = a.runs or sorted(groups)
        os.makedirs(os.path.join(a.out, "sweep"), exist_ok=True)
        for name, fn in [("by_world", sweep_by_world), ("by_readout", sweep_by_readout)]:
            fig = fn(groups, runs, a.panels)
            fig.savefig(os.path.join(a.out, "sweep", f"{name}.png"), dpi=150)
            plt.close(fig)
        print("saved to", os.path.abspath(os.path.join(a.out, "sweep")))
        return

    folder = "Latent structure" if a.kind == "world" else "Uncertainty"
    os.makedirs(os.path.join(a.out, folder), exist_ok=True)
    for path in a.paths:
        saved = torch.load(path, map_location="cpu", weights_only=False)
        fig = world_figure(saved) if a.kind == "world" else readouts_figure(saved, path, a.panels)
        out = os.path.join(a.out, folder, os.path.basename(path)[:-3] + ".png")
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print("saved", out)


if __name__ == "__main__":
    main()
