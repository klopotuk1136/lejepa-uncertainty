import math

import numpy as np
import torch

from lejepa_id.data import ou_augment, sample_latents


class World:
    """Base class. Subclasses set `rho`, `source_dist`, `tag` and implement the two methods."""

    rho = 0.95
    source_dist = "gaussian"   # long-run distribution of z: "gaussian", "laplace" or "gennorm"
    source_alpha = None        # shape parameter, only used by "gennorm"
    is_paper_world = False     # True only where the paper's Theorem 3 bound applies

    def sample_states(self, n, device="cpu"):
        return sample_latents(n, 2, dist=self.source_dist, device=device, alpha=self.source_alpha)

    def sample_next(self, z, n_views=2):
        raise NotImplementedError

    def true_noise(self, z):
        raise NotImplementedError

    def log_det_true_noise(self, z):
        """log det Sigma*(z): one number per state, the quantity readouts are scored against."""
        return torch.linalg.slogdet(self.true_noise(z).double())[1]

    def contrast(self, radius=3.0, n_grid=101):
        """Noise contrast: largest / smallest noise scale over the disc |z| <= radius."""
        g = torch.linspace(-radius, radius, n_grid, dtype=torch.float64)
        z = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
        z = z[z.norm(dim=1) <= radius]
        scale = self.log_det_true_noise(z) / (2 * z.shape[1])          # log of the noise scale
        return math.exp((scale.max() - scale.min()).item())

    def __repr__(self):
        return f"{type(self).__name__}({self.tag}, contrast {self.contrast():.2f}x)"


class PaperWorld(World):
    """z' = rho z + sqrt(1 - rho^2) eta, with eta drawn from the same distribution as z.

    This is the paper's channel, sampled by the paper's own `ou_augment`. With a
    Laplace or gennorm `source` it is the paper's non-Gaussian ablation.
    """

    is_paper_world = True

    def __init__(self, rho=0.95, source="gaussian", alpha=None):
        self.rho, self.source_dist, self.source_alpha = rho, source, alpha
        self.tag = f"paper_rho={rho:g}" + ("" if source == "gaussian" else f"_{source}")

    def sample_next(self, z, n_views=2):
        return ou_augment(z, self.rho, n_views=n_views, dist=self.source_dist, alpha=self.source_alpha)

    def true_noise(self, z):
        eye = torch.eye(z.shape[1], dtype=torch.float64, device=z.device)
        return (1 - self.rho ** 2) * eye.expand(len(z), -1, -1)


class CopulaWorld(World):
    """Each coordinate steps independently. With probability 1 - mix the step is the
    paper's Gaussian step; with probability mix it is a Clayton-copula step.

    Recipe for a Clayton step, per coordinate:
        u  = Phi(z)                                    current state as a uniform number
        u' = (u^-theta (w^(-theta/(1+theta)) - 1) + 1)^(-1/theta),   w ~ U(0, 1)
        z' = Phi^-1(u')                                back to a standard normal

    Both kinds of step have the same correlation rho, so `mix` changes the shape of the
    noise and not its overall size. mix = 0 is exactly the paper's world. Because a
    copula has uniform margins, z' is exactly N(0, 1) for every mix.
    """

    def __init__(self, rho=0.95, mix=1.0):
        if not 0.0 <= mix <= 1.0:
            raise ValueError(f"mix must be in [0, 1], got {mix}")
        self.rho, self.mix = rho, mix
        self.theta = clayton_theta_for(rho)
        self.tag = f"copula_mix={mix:g}"

    def sample_next(self, z, n_views=2):
        # The order of the random draws below is the order the original sampler used,
        # so a seed reproduces earlier training runs exactly. Do not reorder.
        zd = z.double().unsqueeze(0).expand(n_views, *z.shape)
        gauss = self.rho * zd + math.sqrt(1 - self.rho ** 2) * torch.randn_like(zd)
        if self.mix == 0.0:
            return gauss.to(z.dtype)
        log_w = torch.log(torch.rand_like(zd).clamp_min(1e-300))
        clayton = clayton_next(zd, log_w, self.theta)
        if self.mix < 1.0:
            use_clayton = torch.rand_like(zd) < self.mix
            clayton = torch.where(use_clayton, clayton, gauss)
        return clayton.to(z.dtype)

    def true_noise(self, z):
        """Diagonal: each coordinate's variance depends only on that coordinate."""
        return torch.diag_embed(self.conditional_variance(z))

    def conditional_variance(self, z):
        """Var(z'_i | z_i) for every coordinate."""
        zd = z.double().reshape(-1)
        mean_g, var_g = self.rho * zd, torch.full_like(zd, 1 - self.rho ** 2)
        if self.mix == 0.0:
            var = var_g
        else:
            mean_c, second_c = clayton_moments(zd, self.theta)
            mean = (1 - self.mix) * mean_g + self.mix * mean_c
            second = (1 - self.mix) * (var_g + mean_g ** 2) + self.mix * second_c
            var = (second - mean ** 2).clamp_min(1e-12)
        return var.reshape(z.shape)

    def contrast(self, lo=-3.0, hi=3.0):
        var = self.conditional_variance(torch.linspace(lo, hi, 601, dtype=torch.float64))
        return math.sqrt((var.max() / var.min()).item())



# Gauss-Hermite rule: 160 fixed points t_i with weights w_i such that
# E[f(t)] for t ~ N(0, 1) equals sum_i w_i f(t_i) almost exactly.
_GH_T, _GH_W = np.polynomial.hermite_e.hermegauss(160)
_GH_T = torch.tensor(_GH_T, dtype=torch.float64)
_GH_W = torch.tensor(_GH_W / math.sqrt(2 * math.pi), dtype=torch.float64)


def log_expm1(x):
    """log(e^x - 1) for x > 0, accurate for small and large x."""
    return torch.where(x > 20, x + torch.log1p(-torch.exp(-x)),            # large x: no overflow
                       torch.log(torch.expm1(x.clamp(max=20))))            # small x: expm1 keeps digits


def normal_quantile_from_log(log_u):
    """Phi^-1(u) given log u. Near u = 1, u itself rounds to 1.0, so the upper half uses
    the gap 1 - u (computed accurately by expm1) and the symmetry Phi^-1(u) = -Phi^-1(1 - u)."""
    lower = torch.special.ndtri(torch.exp(log_u))
    upper = -torch.special.ndtri(-torch.expm1(log_u))
    return torch.where(log_u < math.log(0.5), lower, upper)


def clayton_next(z, log_w, theta):
    """One Clayton step per entry of z, given the randomness as log w."""
    log_u = torch.special.log_ndtr(z)                                    # log Phi(z)
    log_a = -theta * log_u + log_expm1(-(theta / (1 + theta)) * log_w)   # log of u^-theta (w^... - 1)
    log_u_next = -torch.logaddexp(log_a, torch.zeros_like(log_a)) / theta  # log of (a + 1)^(-1/theta)
    return normal_quantile_from_log(log_u_next)


def clayton_moments(z, theta):
    t, w = _GH_T.to(z.device), _GH_W.to(z.device)
    z_next = clayton_next(z[:, None], torch.special.log_ndtr(t)[None, :], theta)   # (n, 160)
    return (z_next * w).sum(-1), (z_next ** 2 * w).sum(-1)


def clayton_correlation(theta):
    mean_next, _ = clayton_moments(_GH_T, theta)
    return (_GH_T * mean_next * _GH_W).sum().item()


_THETA_CACHE = {}


def clayton_theta_for(rho, tol=1e-6):
    """The Clayton theta whose correlation is rho, by bisection on log theta (~12.66 for 0.95)."""
    if rho not in _THETA_CACHE:
        lo, hi = math.log(0.05), math.log(500.0)
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if clayton_correlation(math.exp(mid)) < rho:
                lo = mid
            else:
                hi = mid
        _THETA_CACHE[rho] = math.exp(0.5 * (lo + hi))
    return _THETA_CACHE[rho]


# =============================================================================

def make_world(cfg):
    """Build a world from a config dict, e.g. {"type": "copula", "rho": 0.95, "mix": 0.75}."""
    kind = cfg["type"]
    if kind == "paper":
        return PaperWorld(cfg.get("rho", 0.95), cfg.get("source", "gaussian"), cfg.get("alpha"))
    if kind == "copula":
        return CopulaWorld(cfg.get("rho", 0.95), cfg.get("mix", 1.0))
    raise ValueError(f"unknown world type: {kind}")
