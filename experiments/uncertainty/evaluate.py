"""
Evaluation: compute every readout map for one trained encoder and score it against the
true noise.

    maps, scores = evaluate(encoder, observer, world, settings)

maps   one value per probe state (the disc |z| <= 3 of a 40 x 40 grid, 1184 states):
       true_noise (log det Sigma*), every readout, plus z, h, Sigma_hat and J
scores per readout: Spearman rho and R^2 against the true noise; plus two diagnostics,
       exact-recovery R^2 and the linearization error

All sampling inside evaluate() starts from `settings["seed"]`, so evaluating the same
encoder twice gives the same numbers.
"""

import math

import numpy as np
import torch

from . import estimators as est

DEFAULTS = {
    "seed": 0,
    "probe_radius": 3.0,     # probe states: a disc, not the square. The square's corners
    "probe_grid": 40,        # (|z| up to 4.2) were almost never seen in training.
    "n_views": 256,          # next states per probe state for the residual
    "view_chunk": 1184,      # probe states whose views are embedded together (memory)
    "n_logged": 100_000,     # logged frames for step_size and density_excess
    "k_neighbours": 200,     # logged frames averaged per probe state by step_size
    "smooth_width": None,    # smoothing window for the Jacobian; None = one noise step
    "n_smooth": 48,
    "fd_eps": 0.1,           # finite-difference step (observers without autograd)
    "max_cond": 1e4,         # exact recovery skips states where J is nearly singular
}

# Readouts that get a map and a score. Each is a sum of estimator maps.
READOUTS = {
    "residual": ["residual"],
    "geometry": ["geometry"],
    "residual + geometry": ["residual", "geometry"],
    "smoothed geometry": ["smoothed_geometry"],
    "residual + smoothed geometry": ["residual", "smoothed_geometry"],
    "encoder-only geometry": ["encoder_geometry"],
    "step size": ["step_size"],
    "density excess": ["density_excess"],
    "step size + density excess": ["step_size", "density_excess"],
}


def probe_states(radius=3.0, n_grid=40):
    """The probe states: an n_grid x n_grid grid on [-radius, radius]^2, kept inside the disc."""
    g = torch.linspace(-radius, radius, n_grid)
    z = torch.stack(torch.meshgrid(g, g, indexing="ij"), dim=-1).reshape(-1, 2)
    return z[z.norm(dim=1) <= radius]


def evaluate(encoder, observer, world, settings=None, logged=None):
    """All readout maps and scores for one encoder.

    logged: optional (x_t, x_next) observations of consecutive frames for step_size and
            density_excess (Reacher passes its prerendered training pairs). If None, they
            are simulated from the world and the observer.
    """
    s = {**DEFAULTS, **(settings or {})}
    was_training = encoder.training
    encoder.eval()
    torch.manual_seed(s["seed"])

    z = probe_states(s["probe_radius"], s["probe_grid"])
    maps = {"z": z, "true_noise": world.log_det_true_noise(z).float()}
    h = est.embed_states(encoder, observer, z)
    maps["h"] = h

    # residual: embed many next states of every probe state
    covs = []
    for i in range(0, len(z), s["view_chunk"]):
        z_views = world.sample_next(z[i:i + s["view_chunk"]], s["n_views"])     # (V, p, d)
        V, p, d = z_views.shape
        h_views = est.embed_states(encoder, observer, z_views.reshape(V * p, d))
        covs.append(est.residual_covariance(h_views.reshape(V, p, -1)))
    maps["sigma_hat"] = torch.cat(covs)
    parts = {"residual": est.residual(maps["sigma_hat"])}

    # geometry: the Jacobian of z -> x -> h, plain and smoothed over one noise step
    maps["J"] = est.jacobian(encoder, observer, z, eps=s["fd_eps"])
    parts["geometry"] = est.geometry(maps["J"])
    width = s["smooth_width"] or math.sqrt(1 - world.rho ** 2)
    if s["n_smooth"]:
        J_smooth = est.smoothed_jacobian(encoder, observer, z, width, s["n_smooth"], eps=s["fd_eps"])
        parts["smoothed_geometry"] = est.geometry(J_smooth)
    if observer.differentiable:
        x = observer.observe(z.to(next(encoder.parameters()).device))
        if x.shape[1:] == z.shape[1:]:                    # encoder Jacobian is square only in 2-D
            parts["encoder_geometry"] = est.geometry(est.encoder_jacobian(encoder, x))

    # logged data: step size and density excess
    if logged is None:
        z_t = world.sample_states(s["n_logged"])
        z_next = world.sample_next(z_t, 1)[0]
        h_t, h_next = est.embed_states(encoder, observer, z_t), est.embed_states(encoder, observer, z_next)
    else:
        h_t, h_next = est.embed_observations(encoder, logged[0]), est.embed_observations(encoder, logged[1])
    parts["step_size"] = est.step_size(h_t, h_next, h, s["k_neighbours"])
    parts["density_excess"] = est.density_excess(h_t, h)

    # combine into readouts and score them
    truth = maps["true_noise"].double()
    scores = {}
    for name, needed in READOUTS.items():
        if all(k in parts for k in needed):
            maps[name] = sum(parts[k].double() for k in needed).float()
            scores[f"rho: {name}"] = spearman(maps[name], truth)
            scores[f"R2: {name}"] = r2_offset_free(maps[name], truth)
    scores.update(exact_recovery(maps["sigma_hat"], maps["J"], world.true_noise(z), s["max_cond"]))

    if was_training:
        encoder.train()
    return maps, scores


# =============================================================================
# Scores
# =============================================================================

def is_constant(x, rtol=1e-6):
    """True if x has no real spread (the paper's world: the same noise everywhere). Then
    every score is undefined. A plain std() test fails here: float32 rounding gives a
    constant field a tiny spread, and scores fitted to it would be pure noise."""
    x = torch.as_tensor(x).double()
    return bool((x.max() - x.min()) <= rtol * x.abs().max().clamp_min(1e-12))


def finite_only(a, b):
    """Both as float arrays, keeping only the states where both are finite. A readout can be
    infinite where the encoder folds space flat (det J = 0)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    return a[keep], b[keep]


def spearman(a, b):
    """Rank correlation: the Pearson correlation of the ranks."""
    a, b = finite_only(a, b)
    if is_constant(b):
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def r2_offset_free(pred, truth):
    """R^2 of pred as an estimate of truth, allowing one additive constant (log det is only
    recovered up to a constant) but no rescaling."""
    pred, truth = finite_only(pred, truth)
    if is_constant(truth):
        return float("nan")
    diff = (pred - pred.mean()) - (truth - truth.mean())
    return float(1 - diff.var() / truth.var())


def exact_recovery(sigma_hat, J, sigma_star, max_cond=1e4):
    """Undo the encoder's warp and compare with the truth.

    exact_recovery_r2:     Sigma*_rec = J^-1 Sigma_hat J^-T; R^2 (offset free) of
                           log det Sigma*_rec against log det Sigma*. States where J is
                           nearly singular (cond > max_cond) are skipped; their share is
                           reported as exact_recovery_kept.
    linearization_error:   median of ||Sigma_hat - J Sigma* J^T|| / ||Sigma_hat||, which
                           tests the identity everything rests on. Small = the noise steps are
                           small enough for the encoder to look linear over them.
    """
    sigma_hat, J, sigma_star = sigma_hat.double(), J.double(), sigma_star.double()
    predicted = J @ sigma_star @ J.transpose(-1, -2)
    lin_err = (sigma_hat - predicted).flatten(1).norm(dim=1) / sigma_hat.flatten(1).norm(dim=1)

    J_inv = torch.linalg.pinv(J, rcond=1e-6)
    recovered = torch.linalg.slogdet(J_inv @ sigma_hat @ J_inv.transpose(-1, -2))[1]
    truth = torch.linalg.slogdet(sigma_star)[1]
    keep = torch.isfinite(recovered) & (torch.linalg.cond(J) < max_cond)
    r2 = r2_offset_free(recovered[keep], truth[keep]) if keep.sum() >= 8 else float("nan")
    return {"exact_recovery_r2": r2, "exact_recovery_kept": keep.double().mean().item(),
            "linearization_error": lin_err.median().item()}
