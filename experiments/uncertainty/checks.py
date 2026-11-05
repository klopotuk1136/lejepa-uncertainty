"""
Sanity checks to run before training anything.

    python -m uncertainty.checks world                 # every copula level + the paper's world
    python -m uncertainty.checks world --mixes 0 1
    python -m uncertainty.checks reacher               # can this machine render and train Reacher?
"""

import argparse
import os
import time

import numpy as np
import torch
from scipy import stats

from .worlds import CopulaWorld, PaperWorld


def check_world(world, n=200_000, seed=0):
    torch.manual_seed(seed)
    z = world.sample_states(n).double()
    z_next = world.sample_next(z, 1)[0].double()
    flat = z_next.flatten().numpy()
    ks_p = stats.kstest(flat[:50_000], "norm").pvalue
    mean, var = flat.mean(), flat.var()
    skew, kurt = stats.skew(flat), stats.kurtosis(flat) + 3

    z_k = z.clone()
    for _ in range(20):
        z_k = world.sample_next(z_k, 1)[0].double()
    drift = abs(z_k.var().item() - 1)
    corr = np.corrcoef(z_next[:, 0], z_next[:, 1])[0, 1]

    states = torch.tensor([[-2.0, -2.0], [0.0, 0.0], [1.0, -1.0], [2.0, 2.0]], dtype=torch.float64)
    empirical = world.sample_next(states, 100_000).double()                       # (100k, 4, 2)
    emp_var = empirical.var(0)
    true_var = world.true_noise(states).diagonal(dim1=-2, dim2=-1)
    noise_err = ((emp_var - true_var).abs() / true_var).max().item()

    ok = (abs(mean) < 0.01 and abs(var - 1) < 0.01 and abs(skew) < 0.05 and abs(kurt - 3) < 0.1
          and ks_p > 0.001 and drift < 0.02 and abs(corr) < 0.01 and noise_err < 0.05)
    print(f"{world.tag:<20s} {world.contrast():6.2f}x | mean {mean:+.4f} var {var:.4f} "
          f"skew {skew:+.3f} kurt {kurt:.3f} KS p {ks_p:.3f} | after 20 steps var off by {drift:.4f} | "
          f"corr {corr:+.4f} | true-noise error {noise_err:.1%}  {'PASS' if ok else 'FAIL'}")
    return ok


def check_reacher():
    from .observers import ReacherCamera
    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL', 'glfw' if os.name == 'nt' else 'egl')} | "
          f"cuda: {torch.cuda.is_available()}")
    try:
        import scipy, sklearn, yaml  # noqa: F401  run_reacher needs them
        camera = ReacherCamera()
    except Exception as e:  # noqa: BLE001 - report whatever broke
        print(f"FAIL: could not load the Reacher renderer: {type(e).__name__}: {e}\n"
              "  install: pip install --no-deps dm_control && pip install absl-py dm-env dm-tree "
              "glfw lxml mujoco protobuf pyopengl pyparsing requests tqdm scikit-learn")
        return False

    a = camera.observe(torch.tensor([[0.3, -0.7], [0.3, -0.7], [0.4, -0.7]]))
    if a.shape != (3, 3, 64, 64) or a.dtype != torch.uint8 or a.float().std() == 0:
        print(f"FAIL: unexpected frames: {tuple(a.shape)} {a.dtype}")
        return False
    if not torch.equal(a[0], a[1]) or torch.equal(a[0], a[2]):
        print("FAIL: rendering is not deterministic, or different angles give the same frame")
        return False
    t = time.time()
    camera.observe(torch.randn(300, 2))
    ms = (time.time() - t) / 300 * 1000
    print(f"renders 64x64 frames, deterministic; {ms:.2f} ms per frame")

    sec_epoch = float("nan")
    if torch.cuda.is_available():
        from lejepa_id.losses import SIGReg, alignment_loss
        from lejepa_id.models import make_cnn_encoder
        dev = torch.device("cuda")
        cnn, sig = make_cnn_encoder(2, device=dev), SIGReg(n_slices=256).to(dev)
        opt = torch.optim.AdamW(cnn.parameters(), lr=3e-3)
        x = torch.randn(2, 256, 3, 64, 64, device=dev)
        for i in range(25):
            if i == 5:
                torch.cuda.synchronize()
                t = time.time()
            h = torch.stack([cnn(x[0]), cnn(x[1])])
            loss = 1e-3 * sig(h) + (1 - 1e-3) * alignment_loss(h)
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        sec_epoch = (time.time() - t) / 20 * (100_000 // 256)
        print(f"training ~{sec_epoch:.0f} s per epoch (100k pairs, batch 256)")
    else:
        print("no CUDA: training would run on the CPU (far too slow for the sweep)")

    # default sweep: 3 worlds x (200k training frames + evaluation renders), 3 seeds x 100 epochs
    renders_eval = 1184 * 128 + 1184 + 4 * 1184
    renders = 10_000 + 3 * 200_000 + 9 * renders_eval
    print(f"expected sweep time: rendering ~{renders * ms / 3.6e6:.1f} h, "
          f"training ~{9 * 100 * sec_epoch / 3600:.1f} h")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    qs = torch.tensor([[-1.5, -1.5], [0, 0], [1.5, 1.5], [3.0, 0], [0, 3.0]])
    frames = camera.observe(qs)
    fig, axes = plt.subplots(1, len(qs), figsize=(2.2 * len(qs), 2.6))
    for ax, q, f in zip(axes, qs, frames):
        ax.imshow(f.permute(1, 2, 0).numpy())
        ax.set_title(f"z = ({q[0]:g}, {q[1]:g})", fontsize=9)
        ax.axis("off")
    os.makedirs("figures/env", exist_ok=True)
    fig.tight_layout()
    fig.savefig("figures/env/reacher_check.png", dpi=120)
    print("example frames: figures/env/reacher_check.png")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["world", "reacher"])
    ap.add_argument("--mixes", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 0.9, 1])
    a = ap.parse_args()
    if a.what == "world":
        worlds = [PaperWorld()] + [CopulaWorld(0.95, m) for m in a.mixes]
        ok = all([check_world(w) for w in worlds])
    else:
        ok = check_reacher()
    print("\nALL CHECKS PASS" if ok else "\nSOME CHECKS FAILED")


if __name__ == "__main__":
    main()
