# `uncertainty/`: reading uncertainty off a LeJEPA encoder

The paper (Klindt, LeCun & Balestriero) assumes the noise in the world is the same
everywhere. This package trains LeJEPA in worlds where it is not, and tests readouts
that recover how noisy each state is from the trained encoder.

The paper's code (everything outside this folder) is used as is. The only change to it
is one argument, `transition_fn`, in `lejepa_id/engine.py`; without it the engine
behaves exactly like the original.

Run every command from `experiments/`.

## The pieces

| File | What it holds | Interface |
|---|---|---|
| `worlds.py` | how the state moves, and the true noise | `sample_states(n)`, `sample_next(z, n_views)`, `true_noise(z)` |
| `observers.py` | how a state becomes an observation | `observe(z)`, `differentiable` |
| `estimators.py` | the readouts | functions returning one value per probe state |
| `evaluate.py` | runs every readout and scores it against the true noise | `evaluate(encoder, observer, world)` |
| `run.py` | config -> train (paper's code) -> evaluate -> save | command line |
| `reevaluate.py` | evaluates old runs (from the old `run.py`) with this code | command line |
| `plots.py` | pictures from saved results | command line |
| `checks.py` | sanity checks before training | command line |
| `configs/` | `copula.yaml` (2-D, the 150-run sweep), `reacher.yaml` | |

**Worlds.** `PaperWorld`: the paper's channel, z' = ρz + √(1−ρ²)η. `CopulaWorld(rho, mix)`:
each coordinate takes the paper's step with probability 1 − mix and a Clayton-copula step
with probability mix. Clayton noise is small for low z and large for high z; both steps
have correlation ρ, and z' stays exactly N(0, 1).

| mix | 0 | 0.25 | 0.5 | 0.75 | 0.9 | 1 |
|---|---|---|---|---|---|---|
| noise contrast | 1.0 (paper's world) | 2.3 | 3.2 | 4.2 | 5.6 | 13.1 |

**Observers.** `Mixing2D` (spiral, banana, sinusoid), `CouplingFlow` (the paper's "nvp"),
`ReacherCamera` (64×64 frames of the DMC Reacher arm at joint angles z). All 2-D mixings
preserve area.

**Readouts.** Each estimates log det Σ*(z), the true noise, up to a constant:

| readout | needs | idea |
|---|---|---|
| `residual` | many next states from the same state (a simulator) | spread of the next state's embedding |
| `geometry` | the observation function (to differentiate z → x → h) | how much the encoder squeezes space around z |
| `smoothed geometry` | same | the Jacobian averaged over one noise step |
| `encoder-only geometry` | the encoder only (2-D) | dh/dx instead of dh/dz; equal to `geometry` when the mixing preserves area |
| `step size` | a log of consecutive frames | how far the embedding moves per step near z |
| `density excess` | embeddings of observed frames | squeezed regions pile up: denser than a Gaussian |

Combinations are sums: `residual + geometry` (the main readout), `residual + smoothed
geometry`, `step size + density excess` (needs neither a simulator nor the mixing).
Scores: Spearman ρ and R² (offset free) against log det Σ*, on the probe states (a 40×40
grid on [−3, 3]², inside the disc |z| ≤ 3). Diagnostics: `exact_recovery_r2`
(Σ* = J⁻¹ Σ̂ J⁻ᵀ) and `linearization_error` (‖Σ̂ − J Σ* Jᵀ‖ / ‖Σ̂‖).

## Commands

```bash
python -m uncertainty.checks world                     # the sampler keeps z' ~ N(0,1), true noise is right

# one run, or the whole sweep of a config (finished runs are skipped)
python -m uncertainty.run --config uncertainty/configs/copula.yaml --run spiral --world mix=0.75 --seed 1337
python -m uncertainty.run --config uncertainty/configs/copula.yaml --sweep --list
python -m uncertainty.run --config uncertainty/configs/copula.yaml --sweep

# old runs from the previous code
python -m uncertainty.reevaluate --old "results/copula (2)/copula" --out results/uncertainty/reevaluated

# pictures
python -m uncertainty.plots world    results/uncertainty/copula/spiral_copula_mix=0.75_seed=1337.pt
python -m uncertainty.plots readouts results/uncertainty/copula/spiral_copula_mix=0.75_seed=1337.pt
python -m uncertainty.plots readouts <file.pt> --panels "step size" "density excess" "step size + density excess"
python -m uncertainty.plots sweep    results/uncertainty/copula --runs spiral banana sinusoid nvp_mlp
```

Reacher (needs `dm_control` and an OpenGL context; on Windows a desktop session):

```bash
python -m uncertainty.checks reacher
python -m uncertainty.run --config uncertainty/configs/reacher.yaml --sweep
```

## What a run saves

`<out>/<run>_<world tag>_seed=<seed>.json`: settings, identifiability metrics (`r2_hz`,
...), every score (`"rho: residual + geometry"`, `"R2: ..."`, `exact_recovery_r2`, ...),
training curves.

`<same name>.pt`: `result` (the json contents), `encoder_state`, `maps` (z, h, true_noise,
every readout, sigma_hat, J on the probe states), `plot` (100k points z, x, h).

## Adding things

- **A world:** subclass `World` in `worlds.py`, implement `sample_next` and `true_noise`
  (the full d×d covariance), add it to `make_world`. Run `checks world` on it.
- **An observer:** subclass `Observer` in `observers.py`, add it to `make_observer`.
- **A readout:** write the estimator in `estimators.py`, compute it in `evaluate()`, and
  name the combination in `READOUTS`.

## Reproduction of the old code

- Same sampler: for the same seed `CopulaWorld` draws bit-identical samples to the old
  `ClaytonMix`, with the same θ and the same true noise.
- Same training: a run through `uncertainty.run` produces bit-identical encoder weights to
  the old `run.py` (tested at 200 steps on the CPU, spiral and NVP).
- Same evaluation: `reevaluate.py` on all 150 old runs reproduces the saved embeddings and
  Jacobians to float precision; the residual maps agree with the old ones as closely as two
  new random draws agree with each other.
