"""
Observers: how a hidden state z becomes the observation x the encoder sees.

Every observer provides

    observe(z)        observations for a batch of states (torch tensor in, torch tensor out)
    differentiable    True if observe() can be differentiated with autograd. If False, the
                      geometry readout uses finite differences instead (Reacher's renderer).

Observers so far:
    Mixing2D       the paper's hand-made 2-D maps: spiral, banana, sinusoid. All three
                   preserve area (det dx/dz = 1).
    CouplingFlow   the paper's random coupling flow ("nvp"); also preserves area.
    ReacherCamera  renders the DMC Reacher arm at joint angles z as 64x64 RGB frames.

To add an observer: subclass Observer, implement observe(), register it in make_observer.
"""

import os

import numpy as np
import torch

from lejepa_id.mixing import MIXINGS_2D, make_coupling_mixing


class Observer:
    differentiable = True
    name = "observer"

    def observe(self, z):
        raise NotImplementedError


class Mixing2D(Observer):
    """The paper's 2-D mixings, by name: spiral, banana, sinusoid."""

    def __init__(self, name):
        self.name, self.fn = name, MIXINGS_2D[name]

    def observe(self, z):
        return self.fn(z)


class CouplingFlow(Observer):
    """The paper's random coupling flow. In 2-D each layer adds tanh(w * other coordinate)
    to one coordinate, with w = +2 or -2 drawn from `seed`.
    """

    name = "nvp"

    def __init__(self, n_layers=8, seed=1337, device="cpu", signs=None):
        self.n_layers = n_layers
        if signs is None:
            self.fn = make_coupling_mixing(2, n_layers=n_layers, seed=seed, device=device)
        else:
            self.fn = lambda z: _coupling_2d(z, signs)

    def observe(self, z):
        return self.fn(z)

    @classmethod
    def from_saved(cls, z, x, n_layers):
        z, x = torch.as_tensor(z[:2000]).float(), torch.as_tensor(x[:2000]).float()
        best_err, best_signs = float("inf"), None
        for code in range(2 ** n_layers):
            signs = [1.0 if (code >> i) & 1 else -1.0 for i in range(n_layers)]
            err = (_coupling_2d(z, signs) - x).abs().max().item()
            if err < best_err:
                best_err, best_signs = err, signs
        if best_err > 1e-4:
            raise ValueError(f"no sign pattern reproduces the saved flow (error {best_err:.1e})")
        return cls(n_layers=n_layers, signs=best_signs)


def _coupling_2d(z, signs):
    """The paper's coupling flow for 2-D latents, with layer weights 2 * sign."""
    for i, s in enumerate(signs):
        z1, z2 = z[..., :1], z[..., 1:]
        if i % 2 == 0:
            z2 = z2 + torch.tanh(2.0 * s * z1)
        else:
            z1 = z1 + torch.tanh(2.0 * s * z2)
        z = torch.cat([z1, z2], dim=-1)
    return z


class ReacherCamera(Observer):
    """DMC Reacher: the arm drawn at joint angles z (radians), as 64x64 RGB frames (uint8).

    Uses the paper's renderer (prerender.make_env / render_at). The renderer is not
    differentiable, so geometry readouts use finite differences. Rendering needs an
    OpenGL context: MUJOCO_GL=glfw on Windows (desktop session), egl on Linux.
    """

    differentiable = False
    name = "reacher"

    def __init__(self):
        prerender = paper_renderer()
        self._render_at = prerender.render_at
        self._env = prerender.make_env()

    def observe(self, z):
        angles = torch.as_tensor(z).detach().cpu().double().numpy()
        frames = np.stack([self._render_at(self._env, q) for q in angles])
        return torch.from_numpy(frames)        # (n, 3, 64, 64) uint8


def paper_renderer():
    """The paper's prerender module. Its import sets MUJOCO_GL to egl unless it is already
    set, and egl exists only on Linux, so pick the right backend first."""
    os.environ.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "egl")
    import prerender
    return prerender


def make_observer(cfg, seed=1337, device="cpu"):
    """Build an observer from a run config, e.g. {"observer": "spiral"} or
    {"observer": "nvp", "n_layers": 8}."""
    name = cfg["observer"]
    if name in MIXINGS_2D:
        return Mixing2D(name)
    if name in ("nvp", "coupling"):
        return CouplingFlow(n_layers=cfg.get("n_layers", 4), seed=seed, device=device)
    if name == "reacher":
        return ReacherCamera()
    raise ValueError(f"unknown observer: {name}")


def observer_from_saved_run(p):
    """The observer of a saved 2-D run (a dict loaded from its .pt file)."""
    if p["mixing"] in MIXINGS_2D:
        return Mixing2D(p["mixing"])
    return CouplingFlow.from_saved(p["z"], p["x"], p["n_layers"])
