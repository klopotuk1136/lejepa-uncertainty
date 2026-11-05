"""
Train an encoder in a world, evaluate every uncertainty readout, save the results.

Run from experiments/:

    python -m uncertainty.run --config uncertainty/configs/copula.yaml --run spiral --world mix=0.75 --seed 1337
    python -m uncertainty.run --config uncertainty/configs/copula.yaml --sweep          # every job in the config
    python -m uncertainty.run --config uncertainty/configs/copula.yaml --sweep --list   # only show what would run

A config names the world, the training settings, the runs (observer + encoder +
objective) and a sweep (which runs, which world settings, which seeds). Training uses
the paper's own code: lejepa_id.engine for the 2-D worlds, run_reacher for Reacher.

Each run writes two files into the config's `out` folder:
    <name>.json   settings, identifiability metrics, readout scores, training curves
    <name>.pt     the same plus the encoder weights, the evaluation maps and 100k
                  plotting points (z, x, h)
with <name> = <run>_<world tag>_seed=<seed>, e.g. spiral_copula_mix=0.75_seed=1337.
A sweep skips runs whose .json already exists with the same number of training steps.
"""

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from lejepa_id.data import sample_latents
from lejepa_id.engine import train_and_evaluate
from lejepa_id.metrics import compute_all_metrics, compute_recovery_metrics
from lejepa_id.models import make_cnn_encoder, make_matched_encoder, make_mlp_encoder

from .estimators import embed_observations
from .evaluate import evaluate
from .observers import make_observer, paper_renderer
from .worlds import make_world


# =============================================================================
# Building blocks
# =============================================================================

def build_encoder(run_cfg, seed, device):
    """The paper's encoders: a 4 x 256 MLP, the flow-matched encoder, or the Reacher CNN."""
    kind = run_cfg.get("encoder", "mlp")
    if kind == "mlp":
        return make_mlp_encoder(2, hidden=run_cfg.get("hidden", 256), device=device)
    if kind == "matched":
        return make_matched_encoder(2, n_layers=run_cfg.get("n_layers", 4), seed=seed + 77777, device=device)
    if kind == "cnn":
        return make_cnn_encoder(d_latent=2, device=device)
    raise ValueError(f"unknown encoder: {kind}")


class NormalizedEncoder(torch.nn.Module):
    """Reacher: takes uint8 frames, normalizes them like run_reacher.normalize_uint8,
    then applies the CNN. Lets the evaluation feed rendered frames directly."""

    def __init__(self, cnn, mean, std):
        super().__init__()
        self.cnn = cnn
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, frames):
        x = frames.float() / 255.0
        return self.cnn((x - self.mean) / (self.std + 1e-6))


def run_name(run, world, seed):
    return f"{run}_{world.tag}_seed={seed}"


def jobs(cfg):
    """Every (run, world settings, seed) in the config's sweep. A sweep entry such as
    {runs: [spiral, banana], world: {mix: [0, 1]}} expands to all combinations."""
    out = []
    for entry in cfg["sweep"]:
        keys = list(entry.get("world", {}))
        values = [entry["world"][k] for k in keys]
        for run in entry["runs"]:
            for combo in itertools.product(*values):
                for seed in entry.get("seeds", cfg["seeds"]):
                    out.append((run, dict(zip(keys, combo)), seed))
    return out


# =============================================================================
# Training
# =============================================================================

def train_2d(run_cfg, world, train_cfg, seed, device):
    """The paper's 2-D training (lejepa_id.engine), with the world's transition.

    The order of everything that draws random numbers (seed, observer, encoder, eval set,
    training) is the order of the paper's runner, so a seed reproduces earlier runs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    observer = make_observer(run_cfg, seed=seed, device=device)
    encoder = build_encoder(run_cfg, seed, device)
    z_eval = world.sample_states(train_cfg.get("num_eval", 10000), device=device)

    encoder, log = train_and_evaluate(
        encoder, observer.observe, N=2, rho=world.rho,
        lamb=run_cfg.get("lamb"), mode=run_cfg.get("mode", "lejepa"), sigma=run_cfg.get("sigma", 1.0),
        source_dist=world.source_dist, source_alpha=world.source_alpha,
        steps=train_cfg["steps"], batch_size=train_cfg.get("batch_size", 256), lr=train_cfg.get("lr", 3e-3),
        z_eval=z_eval, log_every=train_cfg.get("log_every", 500), device=device,
        transition_fn=world.sample_next)

    # identifiability metrics, as the paper computes them
    encoder.eval()
    with torch.no_grad():
        x_eval = observer.observe(z_eval)
        h_eval = encoder(x_eval)
        h_next = encoder(observer.observe(world.sample_next(z_eval, 1)[0]))
        metrics = compute_all_metrics(z_eval, x_eval, h_eval, h_next, world.rho, 2)
        g = torch.linspace(-3.0, 3.0, 100, device=device)
        z_grid = torch.stack(torch.meshgrid(g, g, indexing="ij"), dim=-1).reshape(-1, 2)
        metrics.update(compute_recovery_metrics(z_grid, encoder(observer.observe(z_grid)), 2, suffix="_grid"))
    if not world.is_paper_world:
        # Theorem 3's bound is derived for the paper's channel only
        metrics.update(delta=float("nan"), D_bound=float("nan"), approx_bound=float("nan"))
    metrics.update(final_align=log["align"][-1], final_sigreg=log["sigreg"][-1],
                   final_whiten=log["whiten"][-1], final_loss=log["total"][-1])
    return encoder, observer, metrics, log, None


def reacher_data(world, cfg):
    """Prerendered training pairs (z_t, z_t+1) for this world, rendered once and cached
    under <data_root>/<world tag>. The paper's eval set goes to <data_root>/eval."""
    prerender = paper_renderer()
    root = Path(cfg["data_root"])
    n_eval = cfg["training"].get("n_eval", 10_000)
    if read_meta(root / "eval").get("n_eval") != n_eval:
        prerender.prerender_eval(argparse.Namespace(data_root=str(root), force=True, n_eval=n_eval, eval_seed=8888))
    out = root / world.tag
    seed = cfg["training"].get("render_seed", 9999)
    n = cfg["training"].get("n_train", 100_000)
    meta = read_meta(out)
    if meta.get("n") != n or meta.get("seed") != seed:          # not cached yet, or other settings
        z_t = np.random.default_rng(seed).standard_normal((n, 2))
        torch.manual_seed(seed)
        z_next = world.sample_next(torch.from_numpy(z_t), 1)[0].numpy()
        env = prerender.make_env()
        print(f"rendering {2 * n} frames for {world.tag} ...")
        img_t, img_next = prerender.render_batch(env, z_t), prerender.render_batch(env, z_next)
        meta = {"world": world.tag, "contrast": world.contrast(), "n": n, "seed": seed}
        prerender.save_dataset(out, z_t.astype(np.float32), z_next.astype(np.float32), img_t, img_next, meta)
    return str(out), str(root / "eval")


def read_meta(folder):
    path = Path(folder) / "meta.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def train_reacher(run_cfg, world, cfg, seed, device):
    """The paper's Reacher training (run_reacher.train_one / final_eval) on this world's pairs."""
    import run_reacher as rr

    data_dir, eval_dir = reacher_data(world, cfg)
    z_t, z_next, img_t, img_next, mean, std, _ = rr.load_dataset(data_dir)
    z_ev, img_ev, _, _ = rr.load_eval(eval_dir)
    dataset = rr.ImageDataset(img_t, img_next, z_t, z_next, mean, std)
    eval_imgs = torch.from_numpy(rr.normalize_uint8(img_ev, mean, std))
    eval_z = torch.from_numpy(z_ev)
    t = cfg["training"]
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=t.get("batch_size", 256), shuffle=True, drop_last=True, pin_memory=True,
        num_workers=t.get("num_workers", 0))   # Windows: each worker would get a copy of ~10 GB

    torch.manual_seed(seed)
    np.random.seed(seed)
    cnn = build_encoder(run_cfg, seed, device)
    n_fast = t.get("n_eval_fast", 2000)
    log = rr.train_one(cnn, loader, (eval_imgs[:n_fast], eval_z[:n_fast]), run_cfg.get("lamb", 1e-3),
                       {"n_slices": t.get("n_slices", 256), "lr": t.get("lr", 3e-3), "epochs": t["epochs"]}, device)
    metrics = rr.final_eval(cnn, dataset.img_t[:10000], dataset.z_t[:10000], eval_imgs, eval_z, device)
    metrics = {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
    metrics.update(final_align=log["align"][-1], final_sigreg=log["sigreg"][-1], final_loss=log["total"][-1],
                   train_mean=mean.tolist(), train_std=std.tolist())

    encoder = NormalizedEncoder(cnn, mean, std).to(device)
    logged = (torch.from_numpy(img_t), torch.from_numpy(img_next))   # consecutive frames, uint8
    return encoder, make_observer(run_cfg), metrics, log, logged


# =============================================================================
# One run
# =============================================================================

def run_one(cfg, run, world_settings, seed, device, steps=None):
    run_cfg = cfg["runs"][run]
    world = make_world({**cfg["world"], **world_settings})
    train_cfg = dict(cfg["training"])
    if steps is not None:
        train_cfg["steps"] = steps
    name = run_name(run, world, seed)
    print(f"\n=== {name}  ({world}) ===")

    t0 = time.time()
    if run_cfg["observer"] == "reacher":
        encoder, observer, metrics, log, logged = train_reacher(run_cfg, world, cfg, seed, device)
    else:
        encoder, observer, metrics, log, logged = train_2d(run_cfg, world, train_cfg, seed, device)
    maps, scores = evaluate(encoder, observer, world, cfg.get("evaluation"), logged=logged)

    # plotting points: z, x, h for the world pictures (Reacher: the paper's eval set, no frames)
    if run_cfg["observer"] == "reacher":
        z_plot = torch.from_numpy(np.load(os.path.join(cfg["data_root"], "eval", "z.npy")))
        x_plot = None
        h_plot = embed_observations(encoder, torch.from_numpy(np.load(os.path.join(cfg["data_root"], "eval", "img.npy"))))
    else:
        z_plot = world.sample_states(100_000)
        x_plot = observer.observe(z_plot.to(device)).cpu()
        h_plot = embed_observations(encoder, x_plot)

    result = {"name": name, "run": run, **run_cfg, "world": {**cfg["world"], **world_settings},
              "world_tag": world.tag, "contrast": world.contrast(), "rho": world.rho, "seed": seed,
              "steps": train_cfg.get("steps"), "epochs": train_cfg.get("epochs"),
              "minutes": (time.time() - t0) / 60, **metrics, **scores}
    save(cfg["out"], name, result, log, encoder, maps, (z_plot, x_plot, h_plot))
    print_summary(result)
    return result


def save(out_dir, name, result, log, encoder, maps, plot):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name + ".json"), "w") as f:
        json.dump(to_json({**result, "log": log}), f, indent=1)
    z, x, h = plot
    torch.save({"result": result, "log": log,
                "encoder_state": {k: v.cpu() for k, v in encoder.state_dict().items()},
                "maps": {k: v.cpu() for k, v in maps.items()},
                "plot": {"z": z.cpu(), "x": None if x is None else x.cpu(), "h": h.cpu()}},
               os.path.join(out_dir, name + ".pt"))


def to_json(obj):
    """numpy / torch scalars and arrays -> plain Python, for json.dump."""
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, (np.generic, torch.Tensor)):
        return obj.tolist() if getattr(obj, "ndim", 0) else obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def print_summary(r):
    print(f"  identifiability R2(h->z) = {r['r2_hz']:.4f}   ({r['minutes']:.1f} min)")
    for key in ("residual", "geometry", "residual + geometry", "step size + density excess"):
        if f"rho: {key}" in r:
            print(f"  rho {key:<28s} {r[f'rho: {key}']:+.3f}")


# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", help="a run name from the config's `runs`")
    ap.add_argument("--world", nargs="*", default=[], metavar="KEY=VALUE",
                    help="world settings on top of the config's, e.g. mix=0.75")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--steps", type=int, help="override the number of training steps (2-D)")
    ap.add_argument("--sweep", action="store_true", help="run every job in the config's sweep")
    ap.add_argument("--list", action="store_true", help="with --sweep: only list the jobs")
    ap.add_argument("--force", action="store_true", help="with --sweep: redo finished runs")
    a = ap.parse_args()

    # the paper's training loop prints "R²"; Windows consoles default to a codepage without it
    sys.stdout.reconfigure(encoding="utf-8")
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not a.sweep:
        world_settings = {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in a.world)}
        run_one(cfg, a.run, world_settings, a.seed, device, a.steps)
        return

    steps = a.steps or cfg["training"].get("steps")
    todo = []
    for run, world_settings, seed in jobs(cfg):
        name = run_name(run, make_world({**cfg["world"], **world_settings}), seed)
        path = os.path.join(cfg["out"], name + ".json")
        done = False
        if os.path.exists(path):
            with open(path) as f:
                done = json.load(f).get("steps") == steps
        print(f"  {'skip' if done and not a.force else 'run '}  {name}")
        if a.force or not done:
            todo.append((run, world_settings, seed))
    print(f"{len(todo)} to run")
    if a.list:
        return
    t0 = time.time()
    for i, (run, world_settings, seed) in enumerate(todo):
        run_one(cfg, run, world_settings, seed, device, a.steps)
        per_run = (time.time() - t0) / (i + 1)
        print(f"[{i + 1}/{len(todo)}] elapsed {(time.time() - t0) / 60:.0f} min, "
              f"ETA {per_run * (len(todo) - i - 1) / 60:.0f} min")


if __name__ == "__main__":
    main()
