import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde, multivariate_normal
from torch.func import jacrev, vmap


@torch.no_grad()
def embed_observations(encoder, x, chunk=4096):
    """h = encoder(x), in chunks so large batches fit in memory. Returns a CPU tensor."""
    device = next(encoder.parameters()).device
    return torch.cat([encoder(x[i:i + chunk].to(device)).cpu() for i in range(0, len(x), chunk)])


def embed_states(encoder, observer, z, chunk=4096):
    """h = encoder(observe(z)), observing one chunk at a time (rendering is slow and big)."""
    device = next(encoder.parameters()).device
    return torch.cat([embed_observations(encoder, observer.observe(z[i:i + chunk].to(device)))
                      for i in range(0, len(z), chunk)])


def jacobian(encoder, observer, z, eps=0.1, chunk=4096):
    """J(z) = dh/dz for the whole map z -> x -> h, shape (n, d, d).

    Differentiable observers use autograd. Others use central differences
    (h(z + eps e_j) - h(z - eps e_j)) / 2 eps along each latent direction e_j.
    """
    device = next(encoder.parameters()).device
    z = z.to(device)
    if observer.differentiable:
        def h_of_z(one_z):
            return encoder(observer.observe(one_z.unsqueeze(0))).squeeze(0)
        return torch.cat([vmap(jacrev(h_of_z))(z[i:i + chunk]).detach().cpu()
                          for i in range(0, len(z), chunk)])
    columns = []
    for j in range(z.shape[1]):
        step = torch.zeros_like(z)
        step[:, j] = eps
        h_plus = embed_states(encoder, observer, z + step, chunk)
        h_minus = embed_states(encoder, observer, z - step, chunk)
        columns.append((h_plus - h_minus) / (2 * eps))
    return torch.stack(columns, dim=-1)


def smoothed_jacobian(encoder, observer, z, width, n_offsets=48, seed=0, eps=0.1):
    """J averaged over a Gaussian window around each state: the mean of J(z + offset) over
    n_offsets random offsets with standard deviation `width`. The same offsets are used for
    every state. With width = one noise step it matches the scale the residual is measured on."""
    g = torch.Generator().manual_seed(seed)
    offsets = torch.randn(n_offsets, z.shape[1], generator=g) * width
    return sum(jacobian(encoder, observer, z + o, eps=eps) for o in offsets) / n_offsets


def encoder_jacobian(encoder, x, chunk=4096):
    """dh/dx of the encoder alone, at observations x. Only square when x has the same size
    as h (the 2-D worlds). Needs no access to how x was made."""
    device = next(encoder.parameters()).device
    x = x.to(device)

    def h_of_x(one_x):
        return encoder(one_x.unsqueeze(0)).squeeze(0)
    return torch.cat([vmap(jacrev(h_of_x))(x[i:i + chunk]).detach().cpu()
                      for i in range(0, len(x), chunk)])


def residual_covariance(h_views):
    """Sigma_hat for each probe state: the covariance of the embeddings of its next states.
    h_views: (n_views, n_probe, d)  ->  (n_probe, d, d)."""
    centred = h_views - h_views.mean(0, keepdim=True)
    return torch.einsum("vpi,vpj->pij", centred, centred) / (len(h_views) - 1)


# Estimators: each returns a map, one estimate of log det Sigma* per probe state

def residual(sigma_hat):
    """2 log tr Sigma_hat. Needs: next states sampled from each probe state."""
    return 2 * torch.log(sigma_hat.diagonal(dim1=-2, dim2=-1).sum(-1))


def geometry(J):
    """-2 log|det J|: large where the encoder squeezes. Needs: J of the full map z -> h.
    float64: where the encoder folds space, det J is so small that float32 rounds it to 0."""
    return -2 * torch.linalg.slogdet(J.double())[1]


def step_size(h_t, h_next, h_probe, k=200):
    """2 log of the mean squared embedding step ||h(x_t+1) - h(x_t)||^2 over the k logged
    frames whose embedding is nearest to each probe embedding. Needs: consecutive frames."""
    squared_steps = ((h_next - h_t) ** 2).sum(1).numpy()
    _, nearest = cKDTree(h_t.numpy()).query(h_probe.numpy(), k=k)
    return torch.from_numpy(2 * np.log(squared_steps[nearest].mean(1)))


def density_excess(h_data, h_probe):
    """2 [log p(h) - log N(h)]: how much denser the embeddings are than the Gaussian fitted
    to them. p is a kernel density estimate of the logged embeddings. Needs: embeddings of
    observed frames, and that the true latents are (close to) Gaussian, which fixes the
    coordinates the noise is measured in."""
    data = h_data.double().numpy()
    probe = h_probe.double().numpy()
    log_p = np.log(gaussian_kde(data.T)(probe.T))
    log_gauss = multivariate_normal(data.mean(0), np.cov(data.T)).logpdf(probe)
    return torch.from_numpy(2 * (log_p - log_gauss))
