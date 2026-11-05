"""
Core training engine — single function used by all experiments.

Handles: LR schedule (warmup + cosine), online data generation,
loss computation (lejepa or whiten), periodic evaluation of ALL metrics
on a fixed eval set, standardized output schema.
"""

import torch
import numpy as np

from .losses import SIGReg, whitening_loss, alignment_loss, infonce_loss
from .data import sample_latents, ou_augment
from .metrics import compute_all_metrics


def warmup_cosine_lr(step, total_steps, base_lr):
    """Constant for first half, cosine decay for second half."""
    warmup = total_steps // 2
    if step < warmup:
        return base_lr
    t = (step - warmup) / (total_steps - warmup)
    return base_lr * 0.5 * (1 + np.cos(np.pi * t))


def train_and_evaluate(
    encoder,
    mix_fn,
    *,
    N,
    rho,
    lamb,
    sigma=1.0,
    mode="lejepa",
    source_dist="gaussian",
    source_alpha=None,
    steps=20000,
    batch_size=256,
    lr=3e-3,
    z_eval,
    log_every=100,
    device="cuda",
    transition_fn=None,
):
    """Train encoder and evaluate periodically.

    Args:
        encoder: nn.Module, x -> h
        mix_fn: callable, z -> x
        N: latent dimension
        rho: OU correlation
        lamb: regularization weight
        mode: "lejepa" or "whiten"
        source_dist: "gaussian", "laplace", or "gennorm"
        steps: total training steps
        batch_size: batch size (online data)
        lr: peak learning rate
        z_eval: (num_eval, N) fixed eval tensor
        log_every: eval frequency
        device: torch device string
        transition_fn: callable (z, n_views) -> (V, B, N). Defaults to the
            homoscedastic OU channel, i.e. the paper's world. The uncertainty
            package passes a world's sample_next (uncertainty/worlds.py).

    Returns:
        encoder: trained encoder
        log: dict of lists — training curves and periodic eval metrics
    """
    if transition_fn is None:
        def transition_fn(z, n_views=2):
            return ou_augment(z, rho, n_views=n_views,
                              dist=source_dist, alpha=source_alpha)

    sigreg = SIGReg().to(device)
    opt = torch.optim.AdamW(encoder.parameters(), lr=lr)

    # Precompute eval mixing (constant across training)
    x_eval = mix_fn(z_eval)

    log_keys = [
        "step", "lr",
        # training losses
        "align", "sigreg", "whiten", "total",
        # eval metrics
        "r2_zx", "r2_xz", "r2_zh", "r2_hz",
        "orth_err", "orth_err_normalized",
        "epsilon", "delta", "D_bound", "approx_bound",
        "procrustes_mse", "L_h", "trace_cov",
    ]
    log = {k: [] for k in log_keys}

    for step in range(steps + 1):
        # LR schedule
        current_lr = warmup_cosine_lr(step, steps, lr)
        for pg in opt.param_groups:
            pg["lr"] = current_lr

        # Online data
        z_batch = sample_latents(batch_size, N, dist=source_dist,
                                 device=device, alpha=source_alpha)
        z_aug = transition_fn(z_batch, 2)                                       # (2, B, N)
        h = encoder(mix_fn(z_aug).flatten(0, 1)).reshape(2, batch_size, N)

        align = alignment_loss(h)
        sig = sigreg(h)
        wht = whitening_loss(h)

        if mode == "lejepa":
            loss = lamb * sig + (1 - lamb) * align
        elif mode == "whiten":
            loss = lamb * wht + (1 - lamb) * align
        elif mode == "infonce":
            loss = infonce_loss(h, sigma)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_every == 0 or (step < 1000 and step % 100 == 0):
            log["step"].append(step)
            log["lr"].append(current_lr)
            log["align"].append(align.item())
            log["sigreg"].append(sig.item())
            log["whiten"].append(wht.item())
            log["total"].append(loss.item())

            # Full eval on fixed set
            encoder.eval()
            with torch.no_grad():
                h_eval = encoder(x_eval)
                z_prime = transition_fn(z_eval, 1).squeeze(0)
                h_prime = encoder(mix_fn(z_prime))

            metrics = compute_all_metrics(z_eval, x_eval, h_eval, h_prime, rho, N)
            
            for k, v in metrics.items():
                log[k].append(v)

            encoder.train()

            if step % (log_every * 10) == 0:
                print(f"  step {step:5d} | lr={current_lr:.1e} "
                      f"align={align.item():.2e} sig={sig.item():.1f} "
                      f"R²(h->z)={metrics['r2_hz']:.4f} "
                      f"orth={metrics['orth_err']:.4f}")

    return encoder, log
