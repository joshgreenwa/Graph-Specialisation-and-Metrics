# Chapter 6 lightweight pilot: local messages, non-local structure

## Decision

**Proceed, with a narrow claim.** The synthetic direction is scientifically
promising because two figures answer two predeclared questions without relying
on fitted-model head interpretations:

1. Can a remote topological edit change RRWP on a locally attended pair while
   leaving its short-horizon features fixed? **Yes.**
2. Can local semantic evidence be selected using that non-local structural
   preprocessing, without transporting the semantic values away from their
   candidate nodes? **Yes.**

This is a capability and information-access result. It does not yet show that
RRWP substitutes for dense communication in GRIT, nor that the same mechanism
explains ZINC performance. Those claims still require the controlled support-by-
horizon factorial in the chapter plan.

## Experimental contract

- Each connected graph has two marked degree-two candidates. One is on a cycle;
  the other is in the interior of a long chain.
- Their rooted radius-two neighbourhoods are exactly isomorphic.
- Candidate values are independent standard-normal variables and the target is
  the value on the cycle.
- A shared local gate scores each candidate from its diagonal RRWP vector and
  predicts a softmax-weighted sum of the two semantic values.
- All horizons use the same 17-channel input width and 18 parameters. Channels
  above the tested horizon are zeroed.
- Because the two candidates are structurally indistinguishable under short
  RRWP, their weights must both be 0.5 and the population MSE is exactly 0.5.
- The registered sweep uses cycle lengths 6, 8, and 10, RRWP horizons 1--16,
  eight paired seeds, and 8,192 held-out semantic draws per seed.

The gate is deliberately simpler than a Graph Transformer. This removes message-
passing and optimisation confounds so the experiment tests information access
directly. It should be described as a controlled local-gating probe, not as a
miniature GRIT benchmark.

## Finding 1: local pair support does not imply local RRWP information

Adding one edge whose nearest endpoint is six hops from the selected node leaves
both the identity and one-step random-walk entries unchanged. Nevertheless:

- the RRWP entry on the selected adjacent bond first changes at order 11; and
- the selected self-pair return probability first changes at order 12.

This directly establishes the chapter's foundational distinction: the support
of learned messages and the dependency scope of structural preprocessing are
different properties.

Figure: `outputs/chapter6_local_nonlocal_synthetic_v1/figures/01_remote_edge_rrwp_sensitivity.png`

## Finding 2: predictive performance follows structural information onset

| Cycle length | First distinguishing RRWP order | K=1 test MSE | K=16 test MSE | Paired MSE reduction (95% CI) | K=16 cycle gate mass |
|---:|---:|---:|---:|---:|---:|
| 6 | 6 | 0.4961 | 0.000022 | 0.4960 +/- 0.0068 | 0.9966 |
| 8 | 8 | 0.5040 | 0.000157 | 0.5038 +/- 0.0023 | 0.9912 |
| 10 | 10 | 0.5018 | 0.001866 | 0.5000 +/- 0.0074 | 0.9695 |

The onset order tracks cycle length exactly rather than occurring at an arbitrary
model horizon. Before that order the probe remains at the theoretical 0.5 MSE
baseline. Once higher-order return probabilities distinguish the candidates, the
same local gate selects the correct semantic value and test error falls sharply.

Figure: `outputs/chapter6_local_nonlocal_synthetic_v1/figures/02_local_semantic_selection_by_rrwp_horizon.png`

## Iteration and robustness finding

A post-run optimisation-budget check used 1,000, 2,500, 5,000, and 10,000 steps.
The **information onset never moved**: it remained at orders 6, 8, and 10. The
exact MSE immediately at onset did depend on the optimisation budget, especially
for the length-10 cycle where the first raw RRWP difference is small. At K=16,
all three tasks approached zero MSE as the budget increased.

Therefore the defensible result is that predictive improvement becomes possible
at the distinguishing horizon and strengthens as more structural signal
accumulates. The exact height of the first post-onset point is not a scientific
estimand and should not be overinterpreted.

## What the figures can support

The figures can confidently support:

> A local attention mask does not bound the information locality of RRWP, and a
> candidate-local computation can use higher-order RRWP to select semantic
> evidence based on remote topology.

They cannot support:

- that dense communication is dispensable on ZINC;
- that multi-hop RRWP substitutes for dense messages in a trained GRIT;
- that ZINC requires non-local structure; or
- that any fitted molecular checkpoint uses the demonstrated algorithm.

## Recommended role in Chapter 6

- Use the remote-edge figure in Section 6.2 as a direct deterministic result.
- Use the horizon sweep in Section 6.3 as a controlled capability proof.
- Keep held-out ZINC performance from the full A--D factorial as the chapter's
  primary endpoint.
- Call the direction a go only if the dense-neutral-pair implementations pass
  their audits and the support-by-horizon result is stable across paired seeds.

The synthetic clears the bar for investing in that factorial. It does not clear
the bar for the chapter's final substitution claim on its own.
