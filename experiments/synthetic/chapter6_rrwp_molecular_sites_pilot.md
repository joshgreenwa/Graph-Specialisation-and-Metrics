# Chapter 6 follow-up: molecular-sites RRWP mediation

## Decision

Making the property half local and half remote bridges the artificial
all-or-nothing performance gap without weakening the causal test. The local
and shuffled-global models are useful and nearly attain their information
limit (`MSE about 0.11`), while useful global RRWP still lets the same
two-layer, one-hop attention architecture recover the property almost exactly
(`MSE about 0.0007`). This is a better primary synthetic than the pure-remote
stress test: raw score distributions, semantic carriage, and head regimes are
now broadly similar even though only global RRWP supports reliable semantic
selection. The remaining differences reveal which diagnostics are genuinely
mechanistic and which are task-dependent.

## Task and fairness contract

Each 37-node molecular-like graph contains three independent reporter sites.
Reporter `i` carries a bounded signed scalar `x_i`. Two length-four arms at the
site are either remotely joined or open, giving the remote state `c_i`. A local
single/triple-bond state `q_i` predicts `c_i` with reliability 0.75 and has
fixed message support in both states. To avoid an artificially all-or-nothing
performance gap, the primary property is explicitly half local and half
remote:

`a_i = (1-w) q_i + w c_i`, `y = kappa * sum_i a_i x_i`, with `w=0.5`.

Here `kappa=[3 E(a_i^2)]^(-1/2)` keeps `Var(y)=1`. At `r=0.75,w=0.5`,
`kappa=4/sqrt(21)` and the optimal value-and-cue-only MSE is `3/28=0.1071`.
The pure remote-selection stress test is retained at `w=1`, where the same
bound is `0.375`. The closure endpoints are five bonds from the reporter, while
the model has two message layers: `5 > 2L = 4`. Local propagation therefore
cannot bring the reporter value and closure state together. Higher-order RRWP
can expose closure without adding nonlocal message edges.

All trained arms use the same 10,065-parameter model, optimizer, sample sizes,
all-node mean pooling, and linear readout:

| Arm | RRWP input | Role |
|---|---|---|
| Local | true `I,P`; orders 2--12 zero | noisy local bond-order cue |
| Global | true `I,...,P^12` | useful local and remote structure |
| Shuffled global | true `I,P`; shuffled orders 2--12 | feature-count/scale control without the true closure association |
| Global at shuffled test input | trained global model, high orders shuffled only at test | causal reliance check, not a fourth trained arm |

The registered intervention protocol is
`molecular_sites_v4_sign_flip_bond_order`; the target mixture is stored
separately in the run configuration. Every seed checkpoint is saved.

## Measurements that answer the mechanism directly

Performance alone shows whether global RRWP is useful, not how it is useful.
The primary mechanistic endpoint is therefore site-selection fidelity. For
each reporter, the value is sign-flipped while all structure and other values
are fixed. Both endpoints lie on the training support. The fitted contribution
is

`Delta_i = 0.5 * [f(x_i) - f(-x_i)]`,

and the known contribution is `tau_i = kappa a_i x_i`. Fidelity is
stratified by whether the noisy local cue is correct or misleading. A model
that genuinely uses remote structure should recover `tau_i` in both strata;
a local-cue model should fail specifically when the cue lies.

Native learned-head scores use the same in-support semantic event and three
RRWP-only structural events at reporter-incident attended pairs: full
(bond-order proxy plus remote closure), proxy only, and remote closure only.
Raw finite responses and response divided by the event's input-space L2 norm
are both retained. The latter is a normalized finite secant, not a separately
dose-matched intervention or infinitesimal derivative.

Final-state Functional carriage is exact because the readout is linear. Every
stored distance bin is checked at runtime to reconstruct the full absolute
mass, while the signed carrier sum reconstructs output movement. Mean
per-event normalized profiles describe geometry; unnormalized means describe
total carriage.

Profile intervals aggregate permutation-exchangeable heads inside each fitted
seed before nested seed/graph resampling. Exact seed/layer/head profiles remain
available as a heatmap. Matched semantic--structural alignment is reported only
after subtracting every other head in the same layer, with the excess first
computed per seed.

## Five-seed findings

The primary run is
`outputs/chapter6_rrwp_molecular_sites_mixed50_s5_v1`. Performance, fidelity,
and alignment half-widths use 95% Student-t inference across the five
independently fitted seeds; profile intervals use the registered nested
seed/graph resampling procedure.

### 1. Global RRWP supplies the missing structural correction

| Evaluation | Test MSE, mean +/- half-width |
|---|---:|
| Local RRWP | `0.112918 +/- 0.001714` |
| Global RRWP | `0.000658 +/- 0.000563` |
| Trained shuffled-global control | `0.112703 +/- 0.002075` |
| Global model, high orders shuffled at test | `0.294886 +/- 0.006311` |

The paired local-minus-global gap is `0.112260 +/- 0.001587`; local and the
trained shuffled control differ by only `0.000215 +/- 0.002282`. The realized
test-set cue-only Bayes MSE is `0.10936`, close to the analytic value `0.10714`.
Thus the local model has essentially exhausted its available information. The
global improvement cannot be attributed to parameter count, input width, or
optimization, and test-time shuffling shows that the trained global model
causally relies on the correct higher-order RRWP field.

The counterfactual site-selection endpoint identifies what performance means.
For correct versus misleading local cues, global fidelity MSE is respectively
`0.000115` and `0.000272`; local fidelity MSE is `0.01209` and `0.113995`.
Global contribution correlations remain above `0.999` in both strata, whereas
the local wrong-cue correlation falls to `0.797`. Global RRWP therefore
improves the quality of structural mediation: it selects the right semantic
value even when the local structural proxy lies.

### 2. Raw scores and their distance geometry can look similar

| Learned-head score | Local raw mean | Global raw mean | Global/local |
|---|---:|---:|---:|
| Semantic | `0.12155` | `0.17426` | `1.43x` |
| Full structural | `0.04572` | `0.05734` | `1.25x` |
| Proxy structural | `0.04572` | `0.05736` | `1.25x` |
| Remote structural | `0` | `0.000248` | -- |

The normalized semantic distance profiles `[d0,d1,d2]` are
`[.174,.654,.171]` locally and `[.126,.632,.242]` globally (total-variation
distance `0.071`). Full-structural profiles are `[.279,.526,.195]` and
`[.319,.554,.127]` (TV `0.068`). Semantic and structural scores co-peak at
distance one in both architectures. Hence neither raw score scale nor a
co-moving distance profile identifies the better mediator by itself.

Event norm is an essential caveat. The global full/proxy structural event is
`4.9x` larger in RRWP input space (`8.31` versus `1.69`). Dividing the finite
response by this norm reverses the structural comparison (`0.02698` local
versus `0.00690` global). These values are finite secants under different
natural events, not dose-matched causal derivatives; both raw and normalized
views must be retained.

### 3. Carriage is broadly similar, but route-specific

Semantic carriage totals are `0.792` local and `0.981` global, with normalized
distance profiles `[.176,.443,.381]` and `[.167,.353,.480]` (TV `0.099`). Full
structural totals are also close (`0.369` and `0.319`), although their geometry
differs more (TV `0.205`). Within each fitted architecture, semantic carriage
tracks the structural route that is actually informative: local semantic
versus proxy-structural TV is `0.041`, while global semantic versus
remote-structural TV is `0.082`. Similar aggregate carriage therefore does not
imply equally accurate structural conditioning.

### 4. Profile variability is a diagnostic, not a universal signature

The mean semantic profile CI width is `0.10397` locally and `0.06064` globally
(`1.71x` wider locally), reproducing the motivating observation. Full
structural widths reverse: `0.10108` locally versus `0.14719` globally. The
global reversal is driven by greater between-seed structural variation; local
within-seed head/event heterogeneity remains larger. It is therefore unsafe to
describe wide local intervals as an intrinsic property of local RRWP. The
effect depends on the measured channel and route mixture.

### 5. Most heads remain relative generalists

Using within-arm, within-channel normalization, local heads are `70%`
generalist, `27.5%` semantic-leaning, and `2.5%` structural-leaning; global
heads are `72.5%`, `17.5%`, and `10%`. Matched-minus-other-head semantic/full-
structural cosine excess is modest in both arms (`0.0398 +/- 0.0261` local,
`0.0463 +/- 0.0195` global), and their paired difference is
`0.0065 +/- 0.0230`. Co-movement is real, but is mostly shared organization and
geometry rather than a special class of pure structural reasoners or a unique
global-head alignment signature.

### Calibration against the pure-remote boundary condition

| Signature | Pure remote (`w=1`) | Mixed (`w=0.5`) |
|---|---:|---:|
| Local/global test MSE | `.393/.00057` | `.113/.00066` |
| Raw semantic global/local | `1.87x` | `1.43x` |
| Raw structural global/local | `0.56x` | `1.25x` |
| Semantic carriage cross-arm TV | `.229` | `.099` |
| Semantic profile-CI local/global | `3.31x` | `1.71x` |
| Structural profile-CI local/global | `1.12x` | `0.69x` |

The mixed task is the better empirical analogue; the pure-remote task remains
a useful boundary condition. Presenting both is more informative than tuning a
single synthetic until every post-hoc feature agrees.

## Scientific conclusion

The strongest cross-task conclusion is behavioural and direct:

> Global RRWP can improve the structural conditioning of semantic information
> under strictly local learned messages; the defensible evidence is a fair
> performance contrast, shuffled higher-order controls, and correct
> counterfactual semantic selection.

The experiment is also a useful falsification test. Similar raw structural
scores and co-peaking distance profiles can coexist with a large and causal
performance difference. Carriage reports where a perturbation travels, not
whether the selected semantic contribution is correct. Wider local intervals
survive for semantic profiles here but not for the full structural channel,
and matched-head alignment does not differ reliably between architectures.
These quantities remain informative diagnostics only with explicit events,
dose accounting, nulls, and seed-level inference.

## Reproduce

```bash
PYTHONPATH=src python -m graph_specialisation_metrics.synthetic.rrwp_molecular_sites \
  --output-dir outputs/chapter6_rrwp_molecular_sites_mixed50_s5_v1 \
  --seeds 0,1,2,3,4 \
  --remote-target-weight 0.5
```

Use `--reanalyze-only` with the same output directory to regenerate tables and
figures from the raw measurements. The focused implementation tests are:

```bash
PYTHONPATH=src pytest -q tests/test_rrwp_molecular_sites.py
```
