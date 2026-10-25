# Reflection models in mirror-array optical IRS for visible light communication

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23647456.svg)](https://doi.org/10.5281/zenodo.23647456)

Simulation code for the paper:

> **Closed-Form Mirror Steering and the Cost of Control Latency in Optical
> IRS-Assisted Visible Light Communication**
> Md. Mahin Rahman, Sadia Tabassum, Mumtazah Mubasshirah, Tahiya Hossain,
> Subaita Nujabah.
>
> *Manuscript in preparation for Optical and Quantum Electronics.* This
> repository is the code and the frozen run behind it; the manuscript itself is
> not published here.

Every figure and every number in the paper is produced by the code here.
`freeze_run.log` is the complete output of one frozen run: each number appears
in it verbatim, under a banner stating the configuration that produced it. Two
kinds of number are the stated exceptions — measured timings, which depend on
the machine, and the metaheuristic gaps, which depend on its floating-point
behaviour — and `verify_paper.py` checks everything else exactly. The timings
the paper quotes come from the earlier runs kept in `archive/`, and
`verify_paper.py` checks those too.

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

## Reproducing and verifying the paper

```bash
git clone <this repo>
cd <this repo>
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python verify_paper.py        # ~20 minutes: every figure, freeze_run.log, then a check
```

`verify_paper.py` runs every experiment and diagnostic, writes
`freeze_run.log`, then checks each number the paper cites against that log and
writes `verification_report.txt`. Expect `60 of 60 cited numbers reproduced`
and `8 of 8 quoted timings found in the archived logs`. To re-check an existing
log without re-running the experiments:

```bash
python verify_paper.py --check    # about a minute
```

`run_freeze.py` regenerates the figures and the log without the check.

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

**Experiments** — each writes one or two figures and prints its own findings.
File numbers follow the order in which the experiments were written, not the
figure numbers in the paper.

| script | figure file | question |
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

**Diagnostics** — separate settings, each stated where the paper cites it

| script | question |
|---|---|
| `physical_c.py` | Is each mirror larger than the ray cone it must pass? (justifies `c = 1`) |
| `aperture_stop.py` | Worst-case aperture margin across the panel sweep |
| `bisector_bound.py` | How far is the closed-form bisector from a locally refined optimum? |
| `noise_regime.py` | Why the model divergence peaks and then declines with array size |
| `crossover.py` | At what surface gain does latency overtake blockage? |
| `check_pose.py` | Regression test for the orientation defect described below |

**Tooling**

| file | purpose |
|---|---|
| `verify_paper.py` | run everything, then check every cited number against the log |
| `run_freeze.py` | run everything without the check |
| `verification_report.txt` | the last check's output |
| `archive/` | the earlier run logs the paper's quoted timings come from |

---

## Headline results

All from `freeze_run.log`.

| | cosine model | specular reflection |
|---|---|---|
| overstatement vs specular | 6.28 dB mean, 8.69 dB max | — |
| overstatement with array size | 4.04 dB at *K* = 36, peak 6.39 dB near *K* = 256, 5.69 dB at *K* = 1600 | — |
| value of knowing the device orientation | **0.00 dB** | **0.00 dB** |
| cost of a stale configuration at 50 ms, no occupant | **0.00 ± 0.00 dB** | 13.30 ± 1.72 dB |
| cost of a stale configuration at 50 ms, mobile occupant | 0.00 dB | 12.99 dB |
| cost of blockage, mobile occupant | 1.22 dB | 1.08 dB |

Neither optimum contains the detector normal: the cosine optimum is `u_out` and
the specular optimum is the bisector of `u_in` and `u_out`, both built from
positions alone. Device orientation therefore cannot change the steering
decision under either model, and knowing it is worth exactly nothing to a
controller whose only decision is the element normal. Under the cosine model,
in addition, the optimum moves so slowly with the receiver that a stale
configuration costs almost nothing — 0.00 dB at 50 ms, and below 0.4 dB out to
a full second.

The overstatement rises with array size while the direct path still dilutes it,
peaks once the reflected path dominates, and then declines as the link moves
toward the shot-noise limit; `noise_regime.py` separates the two effects.

A closed-form bisector update costs 64.9 µs against 11.84 s for a converged GA
in the original frozen run (Linux x86-64), and 39.8 µs against 5.08 s on an
Intel Core i9-14900HX under Windows 11. It recovers 73 % of the zero-latency
bound with no orientation estimate at all. Timings depend on the machine; the
machine-independent statements are that the update is *O(K)* and requires no
search, and that the ratio is about five orders of magnitude. Days 8 and 9
charge the metaheuristic a pinned 7.69 s (`freeze.py`), so their results do not
depend on the machine.

### Correction to an earlier version

An earlier version of this repository reported that knowing the device
orientation was worth 2.24–2.58 dB under specular reflection. That figure was
an implementation artefact. The serving-luminaire choice in `fitness.py` took
`np.argmax` over an array that is exactly zero for elements outside the
detector's field of view, so it silently fell back to luminaire 0 for most of
the array; the orientation-blind controller was being charged for that fallback
index rather than for lacking orientation. With the selection made independent
of the assumed orientation the benefit is 0.00 dB under both models.
`check_pose.py` reproduces the defect and the fix.

---

## Requirements

Python 3.9+, NumPy, Matplotlib, SciPy. See `requirements.txt`. No GPU, no
compilation. The full verification takes about 20 minutes on a laptop CPU; it
was measured on an Intel Core i9-14900HX.

---

## Citing

The manuscript is not yet published, so there is no journal volume, pages or
article DOI to cite. Until there is, cite this repository by its archive DOI:

> Md. Mahin Rahman, Sadia Tabassum, Mumtazah Mubasshirah, Tahiya Hossain,
> Subaita Nujabah. *Reflection models in mirror-array optical IRS for visible
> light communication: simulation code*, version 1.1.0, Zenodo, 2026.
> [doi:10.5281/zenodo.23647456](https://doi.org/10.5281/zenodo.23647456)

`CITATION.cff` carries the metadata and GitHub renders a "Cite this
repository" button from it; its `status:` field is updated as the paper moves
from preparation to submission to publication.

---

## Author

Md. Mahin Rahman — Department of Electrical and Electronic Engineering, Islamic
University of Technology, Gazipur, Bangladesh —
[github.com/thisisdibbo](https://github.com/thisisdibbo) — mahinrahman22@iut-dhaka.edu

## Licence

MIT — see `LICENSE`. The reference paper itself is not covered by this licence
and is not redistributed here.
