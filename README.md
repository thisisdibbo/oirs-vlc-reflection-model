# Reflection models in mirror-array optical IRS for visible light communication

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22917919.svg)](https://doi.org/10.5281/zenodo.22917919)

Simulation code for the paper:

> **On the Reflection Model in Mirror-Array Intelligent Reflecting Surfaces for
> Visible Light Communication: Overstated Gain, Analytic Optima, and the Cost of
> Control Latency**
> Md. Mahin Rahman, Sadia Tabassum, Mumtazah Mubasshirah, Tahiya Hossain,
> Subaita Nujabah. Department of Electrical and Electronic Engineering,
> Islamic University of Technology, Gazipur, Bangladesh.
>
> *Manuscript in preparation, intended for Optics Communications.* This
> repository is the code and the frozen run behind it; the manuscript itself is
> not published here.

Every figure and every number in the paper is produced by the code here, in one
frozen run. `freeze_run.log` is that run's complete output: each figure's
numbers appear in it verbatim, under a banner stating the configuration that
produced them.

---

## What the code does

Two descriptions of a tilted mirror are in simultaneous use in the optical-IRS
literature for VLC:

| | reflected gain depends on | optimal normal |
|---|---|---|
| **cosine** (`model="cosine"`) | `n̂ · u_out` only — the incident direction never constrains the orientation | `n̂ = u_out` |
| **specular** (`model="specular"`) | the law of reflection: the reflected ray must reach the detector | `n̂ ∝ u_in + u_out` |

Both are implemented behind one interface, so every experiment runs twice on
identical geometry, identical noise and identical mobility traces. The only
difference between the two arms is the reflection model.

The reference system reproduced here is Dixit *et al.*, *Scientific Reports*
**15**:27400 (2025), [doi:10.1038/s41598-025-12520-7](https://doi.org/10.1038/s41598-025-12520-7),
which is open access. Its Table 3 specifies the geometry completely enough for
an independent implementation. **The reference paper's PDF is not redistributed
here** — download it from the publisher.

---

## Reproducing the figures

```bash
git clone <this repo>
cd <this repo>
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

sh run_freeze.sh          # ~15 minutes, writes every figure and freeze_run.log
```

For a quick check that the code works, without the frozen sample sizes:

```bash
LIFI_FAST=1 python day5_mobility.py       # ~1 minute
```

```powershell
# PowerShell
$env:LIFI_FAST=1
python day5_mobility.py
Remove-Item Env:\LIFI_FAST
```

**Read the banner.** Every script prints its configuration on the first line.
`FROZEN` means the numbers match the paper. `FAST (development — do NOT use for
the paper)` means they do not.

---

## The frozen configuration

`freeze.py` holds the entire experimental configuration in one module, so that
*which settings produced the figures* has exactly one answer. Do not set a seed
count or a diffuse flag inside a day script; it will be overwritten by the next
freeze run and it will disagree with the paper.

| | frozen | `LIFI_FAST=1` |
|---|---|---|
| mobility traces per cell | 50 | 6 |
| Monte Carlo drops (Day 4) | 10,000 | 2,000 |
| secondary sweeps | 25 | 4 |
| element-sweep inner loop | 12 | 3 |

### Diffuse reflections are not enabled everywhere, deliberately

`compare_models.py`, `validate_optimizers.py`, `sweep_panel.py` and
`sweep_power.py` run with diffuse reflections **off**. They exist to reproduce
the reference system on its own terms, and its p.6 states that it models shot
and thermal noise only. Adding a channel component the original excludes would
turn the reproduction into a comparison against a different system.

Days 4, 5, 6, 8 and 9 are claims about physical behaviour rather than about the
original's numbers, so they run with diffuse **on**. Day 7 sweeps it on and off
by design — that is the experiment.

Note the direction this cuts. Diffuse light needs no steering and cannot go
stale, so it raises the floor a stale IRS falls back to. Enabling it *shrinks*
the staleness penalty (17.4 dB → 15.7 dB at saturation) and therefore works
against this paper's own argument. The frozen setting is the conservative
choice, not the flattering one.

---

## Layout

**Core model**

| file | contents |
|---|---|
| `channel.py` | `Config` (every parameter tagged with its provenance in the reference paper), Lambertian LoS gain, noise, SNR, BER |
| `fitness.py` | `GeometryCache` — both reflection models, and the analytic optimum for each |
| `mirrors.py` | roll/yaw parametrisation of a mirror normal, quantisation, slew time |
| `trace_eval.py` | `TraceEvaluator` — vectorised evaluation over a mobility trace |
| `optimizers.py` | TLBO, GA, PSO behind one interface, charged equal evaluation budgets |
| `mobility.py` | random-waypoint traces with furniture avoidance; correlated orientation |
| `orientation.py` | measured random device orientation (Soltani *et al.* 2019) |
| `blockage.py` | segment–box occlusion by the slab method |
| `diffuse.py` | first-order diffuse bounce (Barry *et al.* 1993) |
| `steering.py` | closed-form bisector control and predictive steering |
| `freeze.py` | **the frozen configuration** |
| `plotting.py` | figure saving and publication typography |

**Experiments** — each writes one or two figures and prints its own findings

| script | figure | question |
|---|---|---|
| `compare_models.py` | 2, 4 | Does the reproduction match? How large is the overstatement? |
| `validate_optimizers.py` | 3 | Is the problem searchable at all, under each model? |
| `sweep_panel.py` | 5 | Does the result survive the unstated panel size? |
| `sweep_power.py` | 12 | Does it survive the calibrated transmit power? |
| `day4_orientation.py` | 6 | What is knowledge of the device pose worth? |
| `day5_mobility.py` | 7 | What does a stale mirror configuration cost? |
| `day6_blockage.py` | 8 | How does blockage compare with latency? |
| `day7_noise.py` | 9 | What do ambient light and diffuse reflections change? |
| `day8_proposed.py` | 10 | Closed-form versus predictive versus metaheuristic control |
| `day9_ber.py` | 11 | BER, and how the two models diverge with array size |

Figure 1 is drawn in TikZ (`fig1_geometry.tex`) and has no script.

---

## Headline results

All from `freeze_run.log`.

| | cosine model | specular reflection |
|---|---|---|
| overstatement vs specular | 6.28 dB mean, 8.69 dB max | — |
| overstatement at *K* = 36 → 400 | 4.04 dB → 6.38 dB | — |
| value of knowing the device pose | **0.00 dB** | 2.24–2.58 dB |
| cost of a stale configuration at 50 ms | **0.00 ± 0.00 dB** | 13.30 ± 1.72 dB |
| cost of blockage, mobile occupant | 1.22 dB | 1.08 dB |

Under the cosine model the optimal normal depends only on mirror and receiver
*positions*. Device orientation therefore cannot change the steering decision,
and a configuration computed a second ago is as good as one computed now. Those
two zeros are structural, not rounding: they are what the model permits.

A closed-form bisector update costs 50.7 µs against 7.69 s for a converged GA
and recovers 73 % of the zero-latency bound with no pose estimate at all. Both
timings are machine-dependent; the machine-independent statement is that the
update is *O(K)* and requires no search.

---

## Requirements

Python 3.9+, NumPy, Matplotlib, SciPy. See `requirements.txt`. No GPU, no
compilation. The frozen run takes about 15 minutes on a laptop CPU; it was
measured on an Intel Core i9-14900HX.

---

## Citing

The manuscript is not yet submitted, so there is no journal volume, pages or
article DOI to cite. Until there is, cite this repository by its archive DOI:

> Md. Mahin Rahman, Sadia Tabassum, Mumtazah Mubasshirah, Tahiya Hossain,
> Subaita Nujabah. *Reflection models in mirror-array optical IRS for visible
> light communication: simulation code*, v1.0.0, Zenodo, 2026.
> [doi:10.5281/zenodo.22917919](https://doi.org/10.5281/zenodo.22917919)

 `CITATION.cff` carries the metadata and GitHub renders a "Cite
this repository" button from it; its `status:` field is updated as the paper
moves from preparation to submission to publication.

## Licence

MIT — see `LICENSE`. The reference paper itself is not covered by this licence
and is not redistributed here.
