# Results log — Li-Fi / OIRS-VLC paper

Extension of: Dixit, V., Kumar, A., Sharan, N., Pandey, S., Kumar, A. & Singh, R. S.,
"Optimizing intelligent reflecting surface assisted visible light communication networks
under blockage and practical constraints using TLBO for IoT applications",
*Scientific Reports* **15**:27400 (2025). DOI 10.1038/s41598-025-12520-7 (open access, CC BY-NC-ND).

Rule for this file: every run gets a dated entry with the seed, what changed, and the
numbers out. One change at a time. Do not trust memory on day 17.

---

## Environment

| Item | Value |
| --- | --- |
| CPU | Intel Core i9-14900HX |
| OS / arch | Windows, AMD64 |
| Python | 3.11, numpy 2.5.3, matplotlib 3.10.x |
| Working dir | `D:\Li-Fi Thesis Paper\lifi` |
| Seed (all runs) | 20260922 |

Report the CPU model in the paper next to every timing number.

---

## 2026-09-22 — Day 2: channel model built, environment verified

First implementation of the Lambertian LoS channel and mirror-array IRS, with
provisional parameters. Superseded by the Day 3 entry below once the base paper's
Table 3 was applied — kept only as a record that the environment reproduces
deterministically.

Cross-platform check: identical deterministic output on Windows/AMD64/numpy 2.5.3 and
Linux/x86_64/numpy 2.4.4. Worth one sentence in the paper's reproducibility statement.

---

## 2026-09-22 — Day 3a: optimisers implemented (TLBO / GA / PSO)

`optimizers.py`, `mirrors.py`, `fitness.py`, `validate_optimizers.py`.

Common interface, per-iteration evaluation counts tracked, because they differ and the
difference matters: **TLBO spends 2 × n_pop evaluations per iteration; GA and PSO spend
n_pop.** Plotting convergence against iteration index alone flatters TLBO.

---

## 2026-09-22 — Day 3b: parameter audit against the base paper

Full read of all 16 pages. Source tags: [T3] = their Table 3, [F1] = Fig. 1,
[p6]/[p7]/[p8] = page text.

### Matched my initial guesses

| Quantity | Value | Source |
| --- | --- | --- |
| Room | 5 × 5 × 3 m | [T3] |
| LED coords S1–S4 | (1.25/3.75, 1.25/3.75, 3.0) | [T3] |
| Semi-half angle Φ₁⁄₂ | 60° | [T3] |
| FoV φ_c | 60° | [T3] |
| PD area A_p | 1 cm² | [T3] |
| Refractive index η | 1.5 | [T3] |
| Modulation | OOK | [p6] |
| Channel structure | h_total = h_LoS + h_IRS, Lambertian | Eq. (5) |

### Changed to match the paper

| Parameter | Was | Now | Source |
| --- | --- | --- | --- |
| Bandwidth B | 20 MHz | **50 MHz** | [T3] |
| Mirror reflectivity δ | 0.95 | **0.9** | [T3] |
| Receiver plane | 0.85 m | **0.80 m** | [F1] H₁ |
| Ambient shot noise | on, I_bg = 740 µA | **off** | [p6] — they model shot + thermal only, ambient treated as a separate +0.49 dB aside |
| IRS geometry | spread across each wall | **compact panels centred at R1 (2.5, 5, 1.5), R2 (0, 2.5, 1.5), R3 (2.5, 0, 1.5), R4 (5, 2.5, 1.5)** | [T3] |
| Angle bounds | ±75° both | **polar β ∈ [0°, 90°], azimuth γ ∈ [0°, 180°]** | [p7] |
| Fitness | mean(SNR_dB) − 50·CoV | **0.7·SNR_linear − 0.3·CoV** | Eq. (15), [T3] weights |

**Fitness units — verified.** Their stated optimum is fitness 6.717536 × 10⁶ at
SNR 69.82 dB, CoV 0.153. Check: 0.7 × 10^(69.82/10) = 6.717 × 10⁶. Exact.
**Their SNR term is linear, not dB.** Using dB changes the optimiser's behaviour entirely.

### Not stated anywhere in the paper

| Missing | Consequence | Handling |
| --- | --- | --- |
| **P_t^Avg** (transmit power) | Everything scales with it | **Calibrated to 1.75 W** so the conventional peak matches their Fig. 3a maximum (~51 dB). Must be disclosed in Table I. |
| **A_IRS** (element area) | Symbol defined in their Table 2, value absent | Does not enter their specular Eq. (1) at all — note this |
| IRS panel extent | Sets IRS peak SNR | `panel_size`, swept below |
| Optical filter gain G_F | — | Set to 1.0 |
| Noise constants I₂, I₃, T, G_ol, C_pd, Γ, g_m | — | Ghassemlooy et al. standard values |
| TLBO population size | — | Not stated. Iterations ≈ 50 (their Fig. 8), 30 independent runs (their Fig. 10) |
| Any mirror angular tolerance | Load-bearing for our result | `accept_floor_deg = 0.5°`, our assumption, swept |

Paper text to write: *"The following parameters are not specified in [ref] and were set to
standard values from Ghassemlooy et al.: ..."*

### Useful admission on their page 8

> "response times of conventional motorized mirror steering units are on the order of
> 10−50 ms, with a switching power of 0.1−0.5 W per IRS element ... These values determine
> realistic bounds for real-time IRS adaptation in VLC systems."

They state the actuator latency, cite it (their refs 33–34), say it bounds real-time
adaptation — and then never use it. This sources our τ_slew from the base paper itself.

### Their stated future work (their Conclusion, verbatim)

> "Future work may explore extending this framework to dynamic environments with user
> mobility and integrating machine learning-based adaptive optimization for real-time IRS
> control and scalability in next-generation IoT networks."

---

## 2026-09-22 — Day 3c: Gate 1 reproduction, both reflection models

`compare_models.py`, 21 × 21 receiver grid, all Table 3 parameters applied.

| Scheme | SNR min | SNR max | mean | CoV(Prx) | mean Prx |
| --- | --- | --- | --- | --- | --- |
| Conventional (no IRS) | 43.68 | 51.03 | 49.10 | 0.191 | 42.99 µW |
| IRS, cosine model (theirs) | 50.72 | 76.46 | 66.45 | 0.772 | 520.52 µW |
| IRS, specular model (ours) | 49.40 | 67.77 | 60.17 | 0.538 | 190.20 µW |

| Check | They report | We get | Verdict |
| --- | --- | --- | --- |
| Conventional band | 43 – 51 dB | **43.68 – 51.03 dB** | matches both ends |
| IRS band | 50 – 84 dB | 50.72 – 76.46 dB | min matches; max 7.5 dB short |

**Gate 1: PASSED** on the conventional anchor. The IRS maximum gap is investigated below.

---

## FINDING 1 — the base paper's reflection model omits the law of reflection

Their Eq. (1) with Eq. (8): mirror orientation enters the channel gain through exactly one
term, cos(α_l,m) = **n̂ · u_out**, the angle between the mirror normal and the direction to
the receiver. The **incoming ray direction never appears**.

A mirror is therefore modelled as a cosine-lobe re-radiator, brightest when aimed at the
user regardless of where the light came from. Under actual specular reflection the
reflected ray 2(u_in·n̂)n̂ − u_in must reach the PD.

| | cosine model (theirs) | specular (physical) |
| --- | --- | --- |
| Optimal normal | n̂ = u_out (aim at user) | n̂ = bisector(u_in, u_out) |
| Depends on LED position? | **no** | yes |
| Closed form? | yes | yes |
| Landscape | smooth | no usable gradient |

### Quantified

**Overstatement: mean 6.28 dB, up to 8.69 dB** across the receiver plane.
Robust to the calibrated transmit power:

| P_t | overstatement mean | max |
| --- | --- | --- |
| 0.50 W | 7.13 dB | 10.41 dB |
| 1.00 W | 6.77 dB | 9.50 dB |
| 1.75 W (calibrated) | 6.38 dB | 8.69 dB |
| 3.00 W | 5.95 dB | 7.96 dB |

**Searchability — random mirror configurations reaching >1% of optimum IRS gain:**

| Model | Hit rate |
| --- | --- |
| cosine | **100.0 %** of draws |
| specular | **0.0 %** of draws |

This is why their TLBO converges beautifully and why the same algorithms fail under
correct physics. A 1 cm² PD at 3 m subtends ~0.1°; random mirrors never land in it.

### Consequence for optimiser choice

Under **both** models the optimum is analytic. Population-based search is the wrong tool
in either case — it is merely *possible* under the cosine model and *impossible* under
specular.

---

## 2026-09-22 — Day 3d: optimiser validation (specular model)

User at (3.60, 1.40, 0.80). LoS-only 50.90 dB, analytic optimum 59.06 dB (+8.16 dB).

**A. Free-angle search**, dim = 2K = 800:

| Alg | best | gap | evals | verdict |
| --- | --- | --- | --- | --- |
| TLBO | 51.07 | 7.99 | 3,630 | infeasible |
| GA | 50.90 | 8.16 | 1,830 | infeasible |
| PSO | 51.05 | 8.01 | 1,830 | infeasible |

Pointing footprint at median mirror distance: **5.3 cm** → tiling the room would need
~8,836 candidate targets. Rules out room-wide target search independently.

**B. Assignment search**, dim = K = 400, 25 local targets at footprint resolution,
equal budget of 45,000 evaluations each:

| Alg | best | gap | t_eval | τ_compute | verdict |
| --- | --- | --- | --- | --- | --- |
| TLBO | 56.49 | 2.58 | 0.109 ms | 4.92 s | stalls (premature convergence) |
| **GA** | **59.06** | **0.00** | 0.110 ms | 4.97 s | converges |
| PSO | 57.41 | 1.65 | 0.109 ms | 4.89 s | partial |

Note: the base paper claims TLBO outperforms GA and PSO. Under the specular model with an
equal evaluation budget, **GA reaches the optimum and TLBO does not.**

### The latency numbers — final, measured on the i9-14900HX

| Quantity | Value |
| --- | --- |
| `t_eval`, single user | **0.109 ms** |
| `t_eval`, room-wide (121-point grid) | **12.45 ms** |
| GA τ_compute to reach the optimum | **4.97 s** |
| Room-wide 30 × 60 TLBO run | **44.8 s** of controller time |

Distance travelled during GA's 4.97 s, room is 5 m across:

| User speed | Distance |
| --- | --- |
| 0.5 m/s | 2.5 m |
| 1.0 m/s | **5.0 m** — the full width of the room |
| 1.5 m/s | 7.4 m |

Analytic bisector: same optimum, O(K) = 400 operations, no search.

Quote these alongside the CPU model. Across three separate runs on this machine
`t_eval` drifted 0.109–0.117 ms (background load); report 0.11 ms and say it is a
mean over runs.

---

## 2026-09-22 — Day 3e: panel-size sweep

`sweep_panel.py`. Panel extent is the one free geometric parameter the paper never states.
**Confirmed on the i9-14900HX — matches the reference run to the decimal.**

| Panel [m] | pitch [mm] | cos max [dB] | spec max [dB] | overstatement [dB] |
| --- | --- | --- | --- | --- |
| 0.50 | 55.6 | 77.71 | 68.91 | 5.10 |
| 0.75 | 83.3 | 77.33 | 68.61 | 5.94 |
| 1.00 | 111.1 | 76.46 | 67.77 | 6.38 |
| 1.50 | 166.7 | 74.38 | 65.75 | 6.40 |
| 2.00 | 222.2 | 72.53 | 64.30 | 6.12 |
| 2.50 | 277.8 | 71.04 | 63.30 | 5.79 |
| 3.00 | 333.3 | 69.19 | 61.79 | 5.55 |
| 4.00 | 444.4 | 66.06 | 60.07 | 5.08 |

**Their 84 dB peak is not reachable at any panel size up to 4 m** (best 77.71 dB).
SNR *falls* as panels grow — spreading mirrors moves most of them off the sweet spot.
Overstatement stays within 5.08–6.40 dB across an 8× size range, so Finding 1 does not
depend on this assumed parameter.

---

## FINDING 2 — their Fig. 6 and Fig. 7 cannot both hold under their stated parameters

Both figures plot the **same four positions** against the **same x-axis** (number of IRS
elements), so they can be compared directly.

**Quote A** — spans their page 10 (last line) into page 11 (first line below Fig. 5):

> "The SNR value increases with the number of IRS elements across all positions, ranging
> from approximately 48–50 *dB* with minimal elements to 85–92 *dB* with 400 elements."

**Quote B** — their page 12, second paragraph below Fig. 7 (search their typo `elemnets`):

> "The received power increases with the number of IRS elements for all positions, ranging
> from nearly zero at minimal elemnets to between 1.4 × 10⁻⁶ W and 5 × 10⁻⁶ W at 400
> elements."

Fig. 7's y-axis carries a ×10⁻⁶ multiplier; Edge-Mid ends at ~5, i.e. 5 µW. Confirmed.

### The anchor agrees, the other end does not

At ~0 IRS elements they report 48–50 dB. Our LoS-only model gives 43.7–51.0 dB at ~43 µW.
**Their noise model and ours agree at that end**, so their own two numbers can be compared
without any assumption of ours in between.

| Their reported Prx | SNR it implies (shot + thermal, B = 50 MHz) |
| --- | --- |
| 1.4 µW | 19.8 dB |
| 5.0 µW | 30.8 dB |
| | *they report 85–92 dB* |

**Shot-noise limit** — zero thermal noise, a perfect photon-limited receiver:
5 µW at 50 MHz gives **52.3 dB**. Their 92 dB is 40 dB above the quantum limit for the
power and bandwidth they state.

Power required for 92 dB under their own noise model: **47.7 mW**, about **9,500×** the
5 µW their Fig. 7 reports.

**The obvious rescue fails.** If Fig. 7 plots IRS-only power while Fig. 6 plots total SNR
(plausible — Fig. 7 starts near zero), then adding 5 µW to 43 µW of LoS raises SNR by
about 1 dB, not 42 dB.

### How to write this — do NOT claim their results are wrong

Neutral, checkable, leaves room for a figure-labelling slip:

> We were unable to reconcile the SNR reported in [ref, Fig. 6] with the received power
> reported in [ref, Fig. 7] under the parameters stated in their Table 3: 5 µW at
> B = 50 MHz corresponds to 30.8 dB under a shot-plus-thermal model, and to 52.3 dB even
> at the shot-noise limit. Both works agree at the LoS anchor (48–50 dB), so we calibrate
> our transmit power to that anchor and report IRS gains relative to it.

---

## 2026-09-22 — tooling note

`plotting.py` v1 caught only `PermissionError`. Windows raised
`OSError [Errno 22] Invalid argument` on a locked PNG (common in OneDrive-synced
folders), which escaped and killed `sweep_panel.py` after the run had finished.
Fixed in v2 by catching `OSError`, and a figure that cannot be written at all is
now reported rather than raised. Numbers printed before a figure write are never
affected by this class of failure.

---

## 2026-09-22 — Day 4: random device orientation

Model from Soltani, Purwita, Zeng, Haas & Safari, IEEE TWC 18(3) 2019
(arXiv:1805.07999); numerical values as tabulated in arXiv:2001.09596.
Elevation: sitting = truncated Laplace μ 41.39° σ 7.68°; walking = truncated
Gaussian μ 29.67° σ 7.78°. Azimuth uniform [0, 2π). 2000 drops, seed 20260922.

Three receiver cases: **face-up** (the base paper's assumption), **genie**
(controller knows the true pose), **blind** (controller assumes face-up).

| Model | Activity | Tilt effect | Pose-blindness | Extra outage |
| --- | --- | --- | --- | --- |
| cosine | sitting | −5.79 dB | **0.00 dB** | +3.4% |
| cosine | walking | −4.99 dB | **0.00 dB** | +1.6% |
| specular | sitting | −5.19 dB | **2.80 dB** | +4.8% |
| specular | walking | −4.42 dB | **2.34 dB** | +2.7% |

### FINDING 3 — orientation-aware control is unmodellable in the base paper's model

Their optimum n̂ = u_out depends only on mirror and receiver POSITIONS, so device
orientation cannot enter the steering decision and pose knowledge is worth
identically zero. Under specular reflection the optimum is the bisector, the LED
direction enters, and pose knowledge is worth 2.3–2.8 dB. Their own conclusion
proposes orientation-aware ML control — which their channel model cannot express.

### Secondary: tilt helps the mean, hurts the tail
Wall-mounted panels at z = 1.5 m mean a face-up PD is badly aimed for wall
reflections. Tilting raises mean SNR ~5 dB but outage goes 0% → 1.6–4.8%, and
0.2–1.4% of drops lose the link entirely (every ray outside the 60° FoV).
Report both or it reads as cherry-picking the mean.

---

## 2026-09-22 — Day 5: mobility traces and the staleness curve

20 seeds × 20 s, dt = 20 ms. Random waypoint + OU-correlated pose (marginals
from `orientation.py`; τ_c = 0.5 s is a modelling choice, swept).
`trace_eval.py` vectorises the per-timestep evaluation — verified against the
per-step path to 1.2 × 10⁻¹¹ dB.

Stale-configuration penalty [dB], mean ± s.d. **(re-run after the Day 6
mobility fix; walkers now deflect around furniture, so these supersede the
first Day 5 table)**:

| Model | Speed | 20 ms | 50 ms | 100 ms | 1000 ms |
| --- | --- | --- | --- | --- | --- |
| cosine | 1.0 m/s | 0.00±0.00 | 0.00±0.00 | 0.01±0.00 | 0.39±0.13 |
| specular | 0.5 | 4.79±1.50 | 11.43±2.42 | 16.37±1.89 | 17.65±1.86 |
| specular | 1.0 | 10.69±2.55 | **14.78±2.48** | 16.95±1.85 | 17.50±1.66 |
| specular | 1.5 | 13.22±2.23 | 15.65±1.57 | 16.76±1.03 | 17.07±0.92 |

Drift from the pre-fix run: the headline case (1.0 m/s, 50 ms) moved 14.82 →
14.78 dB, i.e. not at all. The 0.5 m/s case moved +1.43 dB and 1.5 m/s −0.59 dB,
and **every standard deviation roughly doubled** (1.40 → 2.48 at the headline).
Cause: `make_trace` now defaults to `avoid_furniture=True`, so trajectories
deflect around the Table 3 footprints and differ more between seeds. This is the
correct behaviour — the same walker in the same room, with occlusion modelled or
not — but it means 20 seeds is now marginal. Use ≥ 50 seeds for the figures that
go in the paper.

### FINDING 4 — latency is free in their model, fatal under physics

At τ = 50 ms, the upper end of the actuator response time the base paper itself
states (their p.8), a walking user loses **14.78 ± 2.48 dB** under specular
reflection and **0.00 ± 0.00 dB** under the cosine model. The penalty saturates
near 17 dB — the entire IRS gain, i.e. back to LoS only. They state the actuator
figure, say it "determines realistic bounds for real-time IRS adaptation", and
never use it; under their own model it would not have mattered.

### Robustness to the acceptance assumption
Sweeping `accept_floor` 0.1°–5° (footprint 1.1–52 cm), the 20 ms penalty falls
16.3 → 0.29 dB, so the short-latency value IS assumption-dependent. But at every
acceptance the specular penalty reaches 15.7–17.2 dB by 1 s while the cosine
model stays below 0.4 dB. Present as a sweep, never a single value.

### Methodological note
Single traces gave 6.5 and 11.5 dB for identical settings; the 20-seed mean is
9.87 ± 1.47. Never report a single trace.

---

## 2026-09-22 — Day 6: blockage

Obstacles from the base paper's Table 3 as axis-aligned boxes. Three cases:
none / their static furniture / table+sofa with a walking human. All three link
families tested: LED→mirror, mirror→PD, LED→PD. 10 seeds × 20 s at 1.0 m/s.

**These are the corrected numbers** — see the two fixes below.

| Model | Obstacles | τ=0 | τ=50 ms | Outage @50 ms |
| --- | --- | --- | --- | --- |
| cosine | none | 71.71 | 71.71 | 0.0% |
| cosine | static | 71.05 | 71.05 | 0.2% |
| cosine | mobile | 70.96 | 70.96 | 2.7% |
| specular | none | 65.07 | 50.73 | 53.4% |
| specular | static | 64.42 | 50.12 | 58.9% |
| specular | mobile | 64.39 | 50.15 | 58.9% |

### FINDING 5 — they optimised the small term

    cosine    blockage 0.75 dB + latency  0.00 dB =  0.75 dB
    specular  blockage 0.68 dB + latency 14.24 dB = 14.92 dB

Their Table 3 furniture costs 0.7 dB: the table (0.75 m) and sofa (0.85 m) sit
essentially at the 0.80 m receiver plane and intercept little; one 1.70 m human
in a 5 × 5 m room is rarely in the way. The control latency they state on p.8
but never model costs 14.2 dB — a factor of twenty. Outage tells the same story:
blockage moves specular 53.4% → 58.9%, while latency alone moved it 0.1% → 53.4%.

Specular standard deviations widen at short latency (±2.2–2.4 dB vs ±0.7 with no
obstacles) because the walker now deflects around furniture and trajectories
differ more between seeds. Report them.

### Two corrections made during this run
1. **Walkers passed through the sofa.** A receiver inside a box has every link
   occluded by construction — a mobility artefact, not blockage. First fix was
   post-hoc exclusion, but that dropped **7.4%** of timesteps concentrated in the
   middle of the room, biasing the spatial sample. Replaced with furniture-aware
   mobility (`make_trace(avoid_furniture=True)`, 15 cm clearance): drop rate
   7.44% → 0.00%, nothing excluded.
2. **−300 dB sentinels dragged the means.** Total link loss is now counted
   separately from the dB statistics.
   Before both fixes blockage appeared to cost 28 ± 21 dB. It costs 0.7 dB.

---

## 2026-09-22 — Day 7: augmented channel (ambient noise + diffuse reflections)

Tests the base paper's p.6 claim that the IRS's directional reflections
"naturally suppress both diffuse multipath components and off-angle ambient
light". Both halves tested rather than repeated.

### A. Ambient shot noise

No lux→watts conversion is attempted: it depends on the source spectrum and the
receiver's optical bandpass filter, neither of which the paper states. Their own
0.12 µW figure is taken and converted, not re-derived.

| Ambient | I_bg | Mean SNR | Cost |
| --- | --- | --- | --- |
| none (their baseline) | 0 | 48.79 | — |
| their p.6 aside, 0.12 µW | 0.065 µA | 48.79 | 0.00 dB |
| moderate indoor | 740 µA | 45.67 | 3.12 dB |
| bright / skylight | 5100 µA | 39.63 | 9.16 dB |

Their claim is **self-consistent** — 0.12 µW really does cost nothing — but it
implies an ambient level ~10,000× below what standard VLC noise models assume
(740 µA). Which is correct depends on the unstated optical filter. Report the
sweep, not a verdict.

### B. First-order diffuse reflections

288 patches over all six surfaces, ρ = 0.8, two-stage Lambertian bounce
(Barry et al.). First bounce only — a stated approximation, the same one the
open-source ray-tracing VLC simulators make.

    LoS      41.65 µW
    diffuse   5.36 µW   (12.9% of LoS)
    SNR      48.79 → 49.87 dB   (+1.09 dB)

12.9% is squarely in the expected range for ρ = 0.8, which is a good check on
the implementation.

### CORRECTION IN THE BASE PAPER'S FAVOUR

Diffuse reflection needs no steering and cannot go stale, so it raises the floor
a stale configuration falls back to:

    specular  no diffuse   65.07 dB → 47.92 dB at tau = 500 ms
              + diffuse    65.25 dB → 49.57 dB

The staleness penalty drops **17.15 → 15.68 dB**. Adding the realism they admit
to omitting makes their position *better*. Say this plainly in the paper — a
reviewer who sees every modelling choice stacked against the base paper stops
trusting the author; one who sees a correction that weakens the headline by
1.5 dB believes the remaining 15.7. Effect on the cosine model is nil
(0.13 → 0.12 dB) because there was no penalty to reduce.

Limitations stated, not hidden: IRS panels are treated as uniformly diffuse wall
behind them (slightly over-counts); furniture neither blocks nor contributes
diffuse paths.

---

## 2026-09-22 — Day 8: the proposed scheme

Every scheme charged its own real latency: τ = τ_compute + τ_slew. Metaheuristics
carry GA's measured 4.97 s (Day 3); closed form carries one O(K) update; both
carry the same actuator slew (10–50 ms, base paper p.8). 10 seeds × 15 s,
dt = 10 ms, specular model.

### Control cost per update, K = 400

| Scheme | Order | Fitness evals | Measured |
| --- | --- | --- | --- |
| closed-form bisector | O(K) | 0 | **33.9 µs** |
| TLBO | O(N_iter N_pop K) | 90,000 | — |
| GA | O(N_iter N_pop K) | 45,000 | 4.97 s |
| PSO | O(N_iter N_pop K) | 45,000 | — |

Ratio closed-form : GA = **146,534×** (on the i9-14900HX).

### FINDING 6 — prediction collapses under realistic tracking error

Gain over the metaheuristic baseline [dB]:

| Slew | σ = 0 | σ = 5 cm | σ = 10 cm | σ = 20 cm |
| --- | --- | --- | --- | --- |
| 10 ms | predict +4.8 | **reactive +10.3** | **reactive +11.6** | **reactive +12.0** |
| 20 ms | predict +11.5 | **reactive +3.6** | **reactive +4.9** | **reactive +5.3** |
| 50 ms | predict +15.1 | predict +0.4 | **reactive +0.8** | **reactive +1.3** |

Predictive steering recovers **100%** of the genie bound with a perfect pose
estimate, **12%** at 5 cm tracking error, **5%** at 10 cm.

**Mechanism.** Pointing footprint is 5.2 cm. Reactive steering's aim point is
stale by v·τ_slew (1 cm at 10 ms); predictive steering's aim point carries the
tracking error σ. Prediction trades a small known error for a larger random one.

### THE DESIGN RULE

    predict only when   v * tau_slew  >  sigma

i.e. only when the staleness removed exceeds the estimation error introduced.
At 1 m/s with a 10 ms actuator that demands sub-centimetre positioning, which
VLC localisation does not deliver.

### REVISED RECOMMENDATION — this supersedes the Day 1 plan

The headline contribution is the **reactive closed form**, not the predictive
scheme the plan was built around: 12.4 dB over the metaheuristic baseline at
10 ms slew, 72% of the genie bound, **with no pose estimate at all** and no
dependence on tracking accuracy. Prediction is a conditional upgrade with a
stated requirement.

Reporting the crossover ourselves, with the rule that generates it, is the
version that survives review — "our predictive scheme wins" invites exactly the
question we would have had no answer to.

Note: σ is applied to both position and velocity, but velocity error contributes
under 3 mm over a 50 ms horizon, so this is effectively a position-error sweep.
State that rather than letting a reviewer derive it.

---

## 2026-09-22 — Day 9: BER and the element-count sweep

8 seeds × 12 s, dt = 10 ms, specular model, slew 10 ms, σ = 5 cm.
OOK on IM/DD: P_b = Q(√SNR). All schemes averaged over the **same** timesteps
(sliced to the metaheuristic's lag) — see the correction note below.

### Transmit power needed for BER = 3.8 × 10⁻³ (HD-FEC)

| Scheme | P_t | Saving vs metaheuristic |
| --- | --- | --- |
| no IRS | 20.00 mW | — |
| metaheuristic | 20.00 mW | 0.00 dB |
| predictive (σ = 5 cm) | 10.36 mW | 2.86 dB |
| **reactive closed form** | **3.86 mW** | **7.14 dB** |
| genie | 2.00 mW | 10.00 dB |

**Under latency the metaheuristic needs exactly the transmit power of having no
IRS at all** (H = 2.372 × 10⁻⁵ against 2.371 × 10⁻⁵ — statistically identical).
This is the Day 5 result restated in the currency optics reviewers use, and it
is a harder sentence than any SNR figure.

### FINDING 7 — the overstatement grows with array size

| K | cosine | specular | gap |
| --- | --- | --- | --- |
| 36 | 55.26 | 50.94 | 4.32 dB |
| 64 | 58.35 | 53.12 | 5.24 dB |
| 100 | 60.98 | 55.16 | 5.82 dB |
| 144 | 63.15 | 57.02 | 6.13 dB |
| 256 | 66.60 | 60.24 | 6.35 dB |
| 400 | 69.19 | 62.87 | 6.32 dB |

The cosine model lets every mirror contribute regardless of orientation, so its
gain scales with K more favourably than physics permits. The reference system's
Fig. 6 extrapolates to 400 elements and reports 85–92 dB. **Any extrapolation to
large arrays made under that model is optimistic by a widening margin** — which
is where much of the literature's enthusiasm for large IRS arrays originates.

### Correction made during this run
The first pass averaged "no IRS" over the full trace and each steered scheme
over its own post-lag window, mixing different samples of the walk. That made
the IRS appear marginally *worse* than no IRS. All schemes are now sliced to the
largest lag. Same class of error as the Day 5 one; worth a standing check
whenever lagged and unlagged series are compared.

---

## 2026-09-23 — Day 10: writing. Figure 1 drawn, paper skeleton wired

`main.tex` (elsarticle, ~35 kB) and `refs.bib` are in the output folder.
Sections 3–6 are written in full; Sections 1–2 and the abstract are outlined
only. Today's work was **Figure 1**, `fig1_geometry.tex`, the one figure the
whole argument rests on.

### Figure 1 is a plan view, not an isometric room — deliberately

The first two drafts were 3-D isometric room schematics. Both failed the same
way, and the reason is worth recording because it will recur if anyone redraws
this figure:

The LED grid sits at (1.25, 1.25, 3) etc.; the panel centres sit at
(2.5, 5, 1.5) etc. The offset between a LED and its nearest panel is exactly
(1.25, 1.25, 1.5) m — and the offset between a LED and the standing person is
(2, 2, ·). Both lie along the room diagonal (1,1,·). Any corner-on axonometric
maps the horizontal plane through x_iso ∝ (x cos a − y cos b), which collapses
that diagonal. Result: **two of the four LEDs render behind two of the four
panels, and a third renders on top of the person.** Verified numerically for
symmetric 30°/30° iso, unequal-angle dimetric (20°/40°, 15°/50°, 10°/64°),
cavalier and cabinet oblique. Separating the LEDs requires cos b < 0.6 cos a,
which forces an angle pair that makes the room look sheared.

So: (a-i) a plan view at 0.90 cm/m, (a-ii) a separate height scale carrying
z = 0, 0.80, 1.5 and 3.0 m. Every coordinate is then stated without occlusion.
The example path LED (3.75, 3.75) → mirror on R4 → UE (3.8, 0.9) was chosen so
that neither hop crosses any obstacle footprint in plan — a plan-view ray that
crosses a footprint is genuinely ambiguous about whether it is blocked.

Panel (b) is the argument itself: the same LED, mirror and UE drawn twice.
Left, the cosine model — n̂ = u_out, plus a grey arrow showing where the light
actually goes under the law of reflection (it misses the UE). Right, the
specular model — n̂ ∝ u_in + u_out with the equal-angle arcs marked. A reader
gets the paper's claim from this panel alone.

### TikZ notes (both cost time; don't rediscover them)

- A `\newcommand` that expands to `(x,y)` does **not** work in a node `at`
  position — TikZ parses the position before expanding. Use
  `\tikzdeclarecoordinatesystem` if a custom projection is ever needed again.
- In a TikZ node, a bare colour option (`black!60`) means `color=black!60`,
  which sets **fill** as well as text, silently overriding an earlier
  `fill=white`. Dimension labels came out as solid grey boxes. Use
  `text=black!60`.

### Changes to `main.tex`

- `\usepackage{tikz}` + `\usetikzlibrary{arrows.meta,calc,positioning,shapes.geometric}`
- The `\todo` in the `fig:geometry` environment replaced by
  `\resizebox{\textwidth}{!}{\input{fig1_geometry}}` and a full caption
- Added the missing `\label{sec:panel}` (Section~\ref{sec:panel} was undefined)

Natural size of the figure is 475 × 200 pt (16.7 × 7.0 cm), hence the
`\resizebox`. Full document builds clean: 14 pages, no undefined references,
no undefined citations.

---

## 2026-09-23 — Day 10b: Sections 1–2 written, bibliography rebuilt

### The literature check changed the framing, and for the better

Before writing Related Work I verified — from the actual equations, not
abstracts — how the OIRS-VLC literature models a tilted mirror. Three families
exist:

- **(A) specular**: reflected direction computed by reflecting `u_in` about
  `n̂`; the path counts only if that ray reaches the PD. Optimum = bisector.
- **(B) two-cosine**: both `n̂·u_in` and `n̂·u_out` appear, but equality of the
  incidence and reflection angles is never imposed.
- **(C) single-cosine**: orientation enters ONLY as `n̂·u_out`. Optimum = `u_out`.

What the check found:

| Paper | Family | Evidence |
|---|---|---|
| Dixit et al. 2025 (base paper) | **C** | its Eq. (1) + Eq. (8), read directly |
| Maraqa & Ngatched 2023, IEEE Photonics J. | **C** | arXiv:2307.10164 Eq. (4)–(5): ω and γ occur in exactly one place, `cos(Φ^k_u)` = normal dotted with the direction to the **user**; `cos(ξ^a_k)` is labelled an incidence angle but is never expressed in ω,γ and is not a function of the optimisation variables |
| Fang et al. 2024 survey, *Photonics* | reproduces **C** | same two-stage ω/γ form, attributed to three sources |
| Sun et al. 2024, IEEE TWC | **A** | arXiv:2302.03893 Eq. (3): no mirror-normal cosine at all; alignment is binary, "only one of the N entries … is non-zero", geometry follows the generalised Snell's law |
| Qian et al. 2021, ICC | **A** | Eq. (14) constructs the normal as `unit(S−R) + unit(Q−R)` — the bisector, explicitly |

Abdelhady et al. 2021 (OJ-COMS), the canonical mirror-array paper, could not be
read: every route to the full text is bot-blocked. Indirect evidence points to
**C** (Maraqa & Ngatched take their model from it), but it is recorded as
undetermined and **no claim is made about it in the paper**.

**Consequence for the paper.** The original framing — "the prevailing model is
wrong" — was too strong and would have been an easy target at review. The
correct treatment is established and the bisector is stated explicitly in
Qian et al. 2021. The defensible, and actually stronger, framing is:

> Two incompatible descriptions are in simultaneous use. The single-cosine one
> is not an isolated slip — it appears in at least two independent papers and is
> reproduced in a 2024 survey. Nobody has held the geometry fixed and measured
> what it costs. That measurement is this paper.

This also sharpens the headline claim. The real damage is not the 6.3 dB. It is
that under family (C) the optimum normal is decoupled from the LED, so pose
knowledge is worth 0.00 dB, staleness costs 0.00 dB, and actuator latency
**cannot be expressed at all**. Dixit et al. state a 10–50 ms mirror response
and never use it; under their own model, using it would have changed nothing.
Hamad et al. (ICC 2026) optimise mirror orientation for a *moving* user by DRL
and likewise assume reorientation is instantaneous.

### Sections 1–2 written

`main.tex` §1 (Introduction, with the five contributions enumerated and both
counterweights stated) and §2 (Related work, five subsections) are complete.
Section labels added so the Introduction can forward-reference the results:
`sec:model`, `sec:method`, `sec:reproduction`, `sec:overstatement`,
`sec:orientation`, `sec:proposedres`, `sec:conclusion`.

Every number in §1 was checked line by line against the Contribution summary
below. One stale citation key (`soltani2020`) was found and fixed.

### `refs.bib` rebuilt: 33 entries, 18 from 2023 onward

Every field confirmed against Crossref, OpenAlex or the publisher's landing
page. Fields that could not be confirmed are **omitted with a comment**, never
guessed. Corrections made to what was previously in the file:

- `maraqa2023` had no page range → now vol. 15, no. 4, pp. 1–11
- The Optics Express prototype paper's `\todo` author list is now complete
  (Zhang, Yang, Liu, Sun, Zhang, Song), DOI 10.1364/OE.521062
- `abdelhady2024oirs` (Xplore doc 10586952) is **IEEE Internet of Things
  Journal** 11(20):33110–33119, not a photonics venue → key `singh2024performance`
- `agyeman2026` author list completed → `agyeman2026ris`
- `soltani2020` was **wrong**: "Measurements-Based Channel Models for Indoor
  LiFi Systems" is by Arfaoui, Soltani, Tavakkolnia, Ghrayeb, Assi, Safari and
  Haas. Its journal volume/pages could not be confirmed from any accessible
  source, so it is cited as the arXiv preprint rather than with invented fields.

New material worth having: two MEMS fast-steering-mirror papers that bracket
the 10–50 ms latency assumption from both sides — 400 ms and 84 ms settling
open-loop, 0.5 ms and 0.4 ms with a shaped drive. That turns τ_slew from an
assumption into a cited design variable.

**Build status:** 20 pages, zero LaTeX errors, zero undefined references, zero
undefined citations, all 33 bibliography entries cited.

---

## 2026-09-23 — Day 15 FREEZE: N_SEEDS = 50, diffuse on, every figure regenerated

**This is the frozen run. Every figure in the paper comes from
`freeze_run.log`. No further simulations.**

### What changed mechanically

New `freeze.py` holds the whole configuration in one place, so "what settings
produced the submitted figures" has exactly one answer. Every day script now
imports its seed count and diffuse setting from there rather than declaring its
own. `LIFI_FAST=1` gives a 2-minute development run; no argument gives the
frozen one.

| | before | frozen |
|---|---|---|
| Day 4 Monte Carlo drops | 2,000 | 10,000 |
| Day 5 seeds | 20 | 50 |
| Day 6, 7, 8 seeds | 10 | 50 |
| Day 9 seeds | 8 | 50 |
| Day 9 element-sweep inner loop | 4 | 12 |
| Day 5 acceptance sub-sweep | N//2 = 10 | 25 |

Total run time 15 minutes for all nine scripts. Diffuse reflections turned out
to be nearly free (0.14 s → 0.15 s per TraceEvaluator build), so the freeze cost
was almost entirely the seed count.

### Diffuse is NOT on everywhere, deliberately

`compare_models.py`, `validate_optimizers.py` and `sweep_panel.py` keep diffuse
**off**. They exist to reproduce the reference system on its own terms, and its
p.6 states it models shot and thermal noise only. Adding a channel component
the original excludes would turn the reproduction into a comparison against a
different system — and `p_led` was calibrated to its Fig. 3a under its own
assumptions. Days 4, 5, 6, 8 and 9 are claims about physics rather than about
the original's numbers, so those run with diffuse **on**. Day 7 sweeps it on and
off by design. This is written into the `freeze.py` docstring so it does not get
"tidied up" later.

Note the direction: diffuse light needs no steering and cannot go stale, so it
raises the floor a stale IRS falls back to. Turning it on **shrinks** the
staleness penalty and therefore works against this paper's own argument. The
freeze is the conservative choice, not the flattering one.

### Reconciliation — every number in the paper re-checked

| Claim | Before | Frozen | Note |
|---|---|---|---|
| Overstatement, mean / max | 6.28 / 8.69 dB | **6.28 / 8.69 dB** | unchanged (diffuse off here) |
| Overstatement vs K | 4.3 → 6.3 dB | **4.04 → 6.38 dB** | 12 seeds per point now |
| Pose knowledge, cosine | 0.00 dB | **0.00 dB** | structural; held at 10,000 drops |
| Pose knowledge, specular | 2.3–2.8 dB | **2.24–2.58 dB** | tightened |
| Latency @ 50 ms, 1 m/s, cosine | 0.00 ± 0.00 | **0.00 ± 0.00** | held |
| Latency @ 50 ms, 1 m/s, specular | 14.78 ± 2.48 | **13.30 ± 1.72** | ↓ 1.5 dB from diffuse floor; s.d. halved by 50 seeds |
| Saturated penalty | ~17 dB | **15.7 dB** | diffuse floor |
| 20 ms acceptance sweep span | 16.33 → 0.29 dB | **15.01 → 0.28 dB** | |
| Blockage, specular, mobile | 0.68 dB | **1.08 dB** | 50 seeds; static alone is 0.65 dB |
| Compound, specular | 0.68 + 14.24 = 14.92 | **1.08 + 12.99 = 14.07** | |
| Compound, cosine | 0.75 + 0.00 = 0.75 | **1.22 + 0.00 = 1.23** | |
| Free-angle metaheuristic gap | 11.7 dB *(stale)* | **7.99–8.16 dB** | the 11.7 was wrong; §5.3 already had 7.99–8.16 |
| GA gap, reduced space | 0.00 dB | **0.14 dB** | |
| TLBO / PSO gap | 2.58 / 1.65 | **2.62 / 1.65** | |
| t_eval, single user | 0.11 ms | **0.169 ms** | machine load |
| GA τ_compute | 4.97 s | **7.69 s** | follows t_eval |
| Closed-form update | 33.9 µs | **50.7 µs** | machine load |
| Speed ratio | 146,534× | **151,795×** | now stated as "≈10⁵", not a precise factor |
| Closed form recovers | 72% | **73%** | |
| Gain over metaheuristic @ 10 ms | 12.39 dB | **11.22 dB** | |
| Prediction recovery, σ = 0 / 5 cm | 100% / 12% | **100% / 12%** | held exactly |
| Diffuse counterweight, saturated | 17.2 → 15.7 dB | **17.36 → 15.67 dB** | |
| Reproduction, conventional | 43.68–51.03 dB | **43.68–51.03 dB** | unchanged |
| Reproduction, IRS cosine | 50.72–76.46 dB | **50.72–76.46 dB** | unchanged |

**Two things to take from that table.** First, the qualitative claims are all
robust: 0.00 dB stayed 0.00 dB, the overstatement stayed ~6.3 dB, the
metaheuristic still fails, prediction still collapses at 5 cm of tracking error.
Second, the timing numbers are the softest thing in the paper — they moved ~50%
on machine load alone. They are now pinned in `freeze.py` (`T_EVAL_MS`,
`TAU_META_S`) so Days 8 and 9 charge the metaheuristic the same latency §5.3
reports, and the paper states the ratio as five orders of magnitude with the
machine-independent O(K) argument alongside it.

**One stale number was caught:** the Related Work section said free-angle search
stalls 11.7 dB below the optimum. The actual measurement, in §5.3 all along, is
7.99–8.16 dB. Fixed.

### One reproducibility hole found and closed

The claim "sweeping transmit power over a six-fold range gives 5.95 to 7.13 dB"
had **no script behind it** — it came from an ad-hoc sweep earlier in the work
that was never committed. A number in the paper that no released script
regenerates is exactly what the data-availability statement promises not to
have, so `sweep_power.py` now produces it and is part of the freeze
(Fig. 12). The measured values confirm the claim exactly: 7.13 dB at 0.5 W
down to 5.95 dB at 3.0 W.

Worth noting what the sweep actually shows. The gap is not flat — it narrows by
1.18 dB as P_t rises, monotonically. That is expected: more optical power moves
the link towards the shot-noise limit, where SNR grows as P_t rather than
P_t², so a fixed ratio of received powers maps to a smaller ratio in decibels.
The paper now says this rather than claiming flatness it does not have.

### A claim that changed direction: total link loss

Pre-freeze, Day 4 reported that 0.2–1.4% of drops lost the link entirely — every
arriving ray outside the 60° field of view. **With diffuse reflections included,
that is 0.0% in every case.** Scattered light reaches a tilted detector even
when no specular path does. The paper now states this, and states which way it
cuts: omitting diffuse reflections flatters the *specular* case here, not the
cosine one, which is the opposite of the direction the omission cuts for the
staleness result.

### Figure quality

`plotting.py` gained `publication_style()`: serif to match Elsevier body text,
9 pt base, consistent grid and line weights, TrueType embedding (`pdf.fonttype
= 42` — Type 3 fonts are a common Elsevier rejection). Every script calls it.
PDF is the copy that ships (vector); PNG at 600 dpi is for quick viewing. All
ten figures regenerated; no stray timestamped copies remain.

### Build

20 pages, zero LaTeX errors, zero undefined references, zero undefined
citations, all 33 bibliography entries cited, all ten figures found.

---

## 2026-09-23 — Day 16: abstract and highlights written. Paper body complete.

The abstract was deliberately written last, after the freeze, because the
claims shifted twice during the work and once more at the freeze itself. It now
carries only frozen numbers: 6.3 / 8.7 dB overstatement, 0.00 dB vs 2.2–2.6 dB
pose value, 0.00 dB vs 13.3 ± 1.7 dB staleness, 1.1 dB blockage against 13.0 dB
latency, 73% recovery.

Five Elsevier highlights added, all within the 85-character limit (76, 80, 78,
78, 84). They are kept in `main.tex` beside the abstract so the two cannot
drift apart; Elsevier wants them as a separate file at submission time, so they
will be copied out then, not retyped.

Two `\todo` items closed: final N_SEEDS (50 traces per cell, 10,000 Monte Carlo
drops, one frozen run, configuration in one module) and the CPU model
(Intel Core i9-14900HX).

**The paper body is now complete.** Sections 1–6, abstract, highlights,
12 figures, 33 references, 20 pages, clean build.

### Three `\todo` items remain, and only Mahin can close them

1. `\ead{}` — institutional email
2. `\address[iut]{}` — department, institution, city, country
3. GitHub URL and Zenodo DOI (two places: the Introduction and the Data and
   code availability section)

A fourth is conditional: the paragraph noting that the Fig. 6 / Fig. 7
discrepancy was raised with the corresponding author is marked to be updated or
withdrawn depending on whether a reply arrives.

### Still to produce for the submission package

- cover letter
- graphical abstract
- suggested reviewers (3–5, no conflicts, ideally authors of the specular-model
  papers: Sun, Mei, Yang, Song, Zhang; Qian, Chi, Zhao, Chaaban; Maraqa,
  Ngatched — note the last two are authors of a single-cosine paper, which is a
  reason to suggest them, not to avoid them)
- similarity check before upload

---

## 2026-09-23 — Day 17: submission package built

Three deliverables, all in the output folder.

**`graphical_abstract.png`** — 1569 x 628 px, exactly 2.5:1, over Elsevier's
1328 x 531 minimum with no cropping. Two cards, the same LED / mirror / UE drawn
twice, and under each the consequence it implies: pose worth 0.00 dB and latency
inexpressible on the left, 2.2-2.6 dB and 13.3 dB on the right, with blockage at
1.1 dB for scale. Drawn in TikZ (`graphical_abstract.tex`) so the numbers stay
editable; a note in the file says to change `freeze.py` and re-run rather than
editing a number there.

**`cover_letter.md`** — states the contribution and, deliberately, states the
limits of the claim in the letter itself: that the paper is not asserting the
field is unaware of the law of reflection, that prior work states the bisector
explicitly, and that the contribution is measuring what the simplification
costs. An editor who reads an overclaim in a cover letter desk-rejects; one who
reads a claim already bounded by its author reads on. Both counterweights are
named in the letter too.

**`submission_checklist.md`** — the three fields only Mahin can fill, five
suggested reviewers with the justification for each, the file manifest, and the
pre-upload checks.

### Two judgement calls in the reviewer list, recorded because they look wrong

1. **Maraqa and Ngatched are suggested even though their paper is classified as
   single-cosine in Section 2.2.** That is the reason to suggest them. If the
   classification is wrong, review is where that should surface, not the
   comments section after publication. The argument survives losing one example
   because the reference system is verified directly from its own equations.

2. **The reference system's authors are NOT suggested.** The paper re-examines
   their work; putting them on the list creates a conflict a desk editor may
   read badly, and it is unfair to them. Cite prominently, email separately,
   leave off the list.

### Remaining

- fill the three fields, delete `\newcommand{\todo}`
- GitHub repo + Zenodo deposit (include `freeze_run.log` -- it is the most
  useful single artefact for a sceptical reviewer)
- similarity check before upload, expecting a moderate score from Section 3,
  which necessarily transcribes the reference system's model

---

## 2026-09-23 — Day 18: release repository scaffolded and verified

`README.md`, `LICENSE` (MIT, chosen by Mahin), `requirements.txt`,
`.gitignore` and `CITATION.cff` written into the project folder. The repo is
ready to push; nothing else is needed before the GitHub + Zenodo step, which is
the last blocker on the final `\todo`.

### Verified against a clean checkout, not assumed

Copied **only** the 29 files the README lists into an empty directory and ran
all nine experiment scripts there in fast mode. All nine passed and produced all
11 figure PDFs. Nothing depends on a file the repo does not ship — which is the
failure mode that makes a released repo useless, and it cannot be caught by
running in the development folder.

### Two details in `.gitignore` worth not undoing

- `*.log` is ignored, but `!freeze_run.log` un-ignores the one log that matters.
  Verified with `git check-ignore`: it comes out tracked. Without that line the
  evidence for every number in the paper would have been silently excluded from
  the release.
- The timestamped figure copies that `save_fig` writes when a PDF viewer holds a
  file open (`fig8_blockage_221841.pdf` and friends) are ignored by pattern, so
  they cannot leak into a release again.

`git status` on the clean checkout stages 52 files: 29 source + 11 figure PDFs +
11 PNGs + `freeze_run.log`.

### The README states the diffuse-flag asymmetry prominently

Four scripts run with diffuse off (reproduction) and five with it on
(extension). That looks like an inconsistency to anyone reading the code
without the reasoning, so the README gives the reason and states the direction
it cuts: enabling diffuse *shrinks* the staleness penalty and works against this
paper's own argument.

### What is left, in order

1. `git init && git add -A && git commit && git remote add origin ... && git push`
2. Zenodo: link the GitHub account, flip the switch for this repo, then cut a
   GitHub **release** — the DOI is minted from the release, not from the push
3. Paste the GitHub URL and Zenodo DOI into `main.tex` (two places) and
   `CITATION.cff` (`repository-code`)
4. Fill email and affiliation; delete `\newcommand{\todo}`
5. Similarity check, then submit

---

## 2026-09-23 — Day 19: figures re-authored at final physical size

Caught on a QC pass over the compiled Overleaf PDF, not by looking at the
figures in isolation, which is why it survived the freeze.

### The defect

Every matplotlib figure was authored 24–34 cm wide and then typeset into a
6.5 in (16.5 cm) text block by `width=\textwidth`. LaTeX therefore scaled each
one by 0.44–0.62, and the 8 pt tick labels reached the page at:

| figure | drawn width | tick size on the page |
|---|---|---|
| Fig. 7 staleness | 34.3 cm | **3.5 pt** |
| Figs. 2, 4, 6, 10, 11 | 29.2 cm | 4.1 pt |
| Figs. 3, 9 | 27.9 cm | 4.3 pt |
| Fig. 12 | 24.4 cm | 4.9 pt |

Elsevier's artwork guidance asks for roughly 7 pt minimum at final size. 3.5 pt
is a production query at proof stage, and before that it is a referee squinting
at the paper's central latency result.

Figure 1 was unaffected: it is TikZ and scales with the document.

### The fix

Figures are now authored at the size they are printed. `figsize` widths set to
6.5 in for full-width figures and 4.03 in for the two placed at
`0.62\textwidth`; `fig4` is now drawn at 3.25 in and its `\includegraphics`
width changed from `0.62` to `0.50\textwidth` to match. Every figure now
typesets at a scale between 0.99 and 1.01.

`publication_style()` retuned for the smaller canvas: base 8 pt, ticks 7 pt,
legends 6.5 pt, line width 1.1. Its docstring now states the assumption and
warns against raising `figsize` without raising the fonts, since that is
exactly what caused this.

**Result: smallest text on the page went from 3.5 pt to 6.9 pt.**

### Layout work the narrower canvas forced

- Fig. 2: panel titles collided. "Conventional VLC (no IRS)" → "Conventional
  (no IRS)", and the two IRS titles shortened.
- Fig. 7: the middle legend had six entries in two columns and overflowed the
  canvas, which is what pushed the saved width to 7.55 in. Now one entry per
  model with a note that markers encode speed.
- Fig. 10: a two-line `suptitle` was wider than the figure and forced it to
  7.46 in. Removed — the caption carries that framing anyway.
- Fig. 4: the "IRS panel" legend sat on the heatmap; moved below the axes.

### Verification

Re-ran the full frozen set. **All ten headline numbers are byte-identical to
the previous freeze** — 6.28 / 8.69 dB overstatement, 13.30 ± 1.72 dB staleness,
2.24–2.58 dB pose value, 1.08 and 0.65 dB blockage, 14.07 dB compound, 4.04 →
6.38 dB across K. This was a rendering change and nothing else, which is what
running from `freeze.py` with fixed seeds guarantees.

Paper rebuilds at **30 pages** (up from 29: the shorter figures reflow the
text), zero errors, zero undefined references or citations.

### One thing NOT to act on

The local build reports 2 Type 3 fonts on the Highlights page. The user's
Overleaf build of the same source reports **zero**. This is a bitmap-font
fallback in this container's TeX Live, not a defect in the source. Check it
again on the Overleaf PDF after re-uploading; that is the build that counts.

---

## Contribution summary (for the abstract and intro)

1. A single-cosine IRS channel model (base paper Eq. 1 + Eq. 8) omits the law
   of reflection: orientation enters only as n̂·u_out. Verified to be shared by
   Maraqa & Ngatched 2023 and reproduced in the Fang et al. 2024 survey, so it
   is a recurring formulation, not one paper's slip. **Overstates SNR by 6.3 dB
   mean, 8.7 dB max.** Do NOT write "prevailing" — Sun et al. 2024 (TWC) and
   Qian et al. 2021 (ICC) both enforce the law of reflection, and Qian states
   the bisector explicitly. We are not the first to write it correctly; we are
   the first to measure what the simplification costs.
2. Their Fig. 6 and Fig. 7 cannot be reconciled under their stated parameters:
   5 µW at 50 MHz gives 30.8 dB, and 52.3 dB even at the shot-noise limit,
   against the 85–92 dB reported. **Reported as a reproduction limitation, not
   an error claim.**
3. Under their model, **orientation-aware control is unmodellable** — pose
   knowledge is worth identically 0.00 dB, against 2.2–2.6 dB under specular.
   Their own future work proposes exactly this.
4. Under their model **control latency is free** (0.00 dB at every latency
   tested); under specular it costs **13.3 ± 1.7 dB** at the 50 ms actuator
   response time they themselves state and never use.
5. They optimised the small term: **blockage 1.1 dB, latency 13.0 dB.**
6. The optimum is analytic in both models, so population search is unnecessary
   in one and infeasible in the other. The reactive closed form is
   **~10^5× cheaper** than GA (50.7 µs vs 7.69 s, both machine-dependent;
   the robust statement is O(K) with no search) and recovers 73% of the
   genie bound.
   Prediction requires tracking accuracy better than v·τ_slew.

Counterweight to state plainly: adding the diffuse reflections they admit to
omitting **improves** their position, cutting the saturated staleness penalty
17.4 → 15.7 dB (and the 50 ms penalty 14.7 → 13.3 dB).

---

## Open items

- [x] Confirm panel sweep on own machine — done, exact match
- [x] Record room-wide `t_eval` — 12.45 ms
- [x] Email corresponding author — **sent 2026-09-22**
- [x] Sweep `accept_floor_deg` — done, Day 5 robustness panel
- [x] Re-run Day 5 after the `mobility.py` change — done, table updated
- [x] Verify Quote A and Quote B directly in the PDF — **done 2026-09-22**,
      both transcriptions confirmed. Finding 2 stands.
- [ ] Raise `N_SEEDS` to ≥ 50 for the final Day 5 and Day 6 figures — the s.d.
      roughly doubled once walkers deflect around furniture
- [ ] Delete stray timestamped figure copies (`fig8_blockage_221841.pdf` etc.)
      before the submission snapshot; close PDF viewers before re-running
- [ ] Decide whether Days 5 and 6 get re-run with `diffuse=True` for the final
      figure set, or whether diffuse stays a separate Day 7 result. Re-running
      is more honest but shifts every number by ~1–1.5 dB.
- [x] Day 8 — done; contribution reframed, see Finding 6

### Remaining experimental work is now thin

The plan's Days 9–11 (latency layer, predictive steering, code freeze) are
already done — the latency layer landed on Day 5 and the predictive scheme on
Day 8. What is actually left:

- [x] BER vs transmit power for the surviving schemes (plan's Fig. 8) — `fig11_ber`
- [x] Ablation bar chart: reactive vs predictive vs metaheuristic — `fig10_proposed`
- [ ] Day 15 freeze: `N_SEEDS = 50`, `diffuse=True` throughout, regenerate every
      figure at publication quality, then no new simulations

### Start writing

Against the original 20-day plan we are ~5 days ahead. The plan's own stated
risk was running out of time at the writing end, so that buffer should be spent
on Sections III–V now rather than on more experiments. Every equation needed for
the System Model section is already in `channel.py`, `fitness.py` and
`steering.py` docstrings with its source tagged.

## Superseded / do not reuse

- `reproduce.py` — replaced by `compare_models.py`
- `i_bg = 5100 µA` — costs ~8 dB, puts conventional below their band
- `i_bg = 740 µA` — correct for a lit room, but the paper models no ambient at all
- Mirrors spread across whole walls — Table 3 gives compact panel centres
- Fitness in dB — their Eq. (15) uses linear SNR
- Per-timestep `GeometryCache` in trace loops — use `trace_eval.TraceEvaluator`
- Post-hoc exclusion of timesteps inside furniture — use `avoid_furniture=True`
- Day 6 numbers from before the mobility fix (28 ± 21 dB blockage cost) — wrong
- Single-trace staleness results — always average over seeds
- **Every pre-freeze number (anything logged before the Day 15 FREEZE entry).**
  The day-by-day entries above are kept as a record of what was measured when,
  not as a source of numbers. The paper cites the freeze only. Specifically
  superseded: 14.78 ± 2.48 dB staleness, 0.68 dB blockage, 2.3–2.8 dB pose
  value, 146,534× speed ratio, t_eval = 0.11 ms, tau_compute = 4.97 s,
  33.9 µs closed-form update, 72% recovery, 12.39 dB gain over metaheuristic
- "Free-angle search stalls 11.7 dB below the optimum" — never correct;
  the measurement is 7.99–8.16 dB
- Editing N_SEEDS or a diffuse flag inside a day script — `freeze.py` owns both,
  and a local edit will be silently overwritten by the next freeze run
