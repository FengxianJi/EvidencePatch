# EvidencePatch implementation and first-experiment controls

**Public release entry point:** see [the repository README](../README.md) and [running methods](../docs/RUNNING_METHODS.md). The notes below preserve original experiment context; historical workspace paths are not included raw datasets. Use `scripts/run_method.py` for the supported public CLI.

EvidencePatch is the project's core method family. This folder separates its implementation variants from the ten registered baseline/control conditions below; the ten controls are not ten proposed methods. See the project-level [release scope](../docs/RELEASE_SCOPE.md) for the current implementation state and evidence limits.

This package is scoped to `/show/experiment/[1]first experiment`. It supplies ten registered controls through `method.get_method(method_id)(context)`. Every call returns a dictionary with `method_id` and `appendix_markdown`; methods may add diagnostics such as selected actions, verification labels, unresolved concerns, and API-call count.

## Ten registered baselines and control conditions

| ID | Comparison role | API calls | Use |
|---|---|---:|---|
| `frozen_original` | unchanged manuscript | 0 | primary no-intervention control |
| `one_shot_generic` | generic M3 appendix | 1 | direct LLM baseline; wired to the existing v4 development comparison |
| `iterative_generic` | generic draft then revision | 2 | iteration ablation; needs verifier feedback in context to be informative |
| `drpg_style` | decompose/retrieve/plan/generate prompt | 1 | related-work-inspired pipeline comparator, not a reproduction |
| `retrieval_only` | evidence pointers only | 0 | retrieval ablation; no entailment claim |
| `peerify_verify` | claim-level verification | 1 | related-work-inspired diagnostic; does not itself generate a patch |
| `random_budget` | seeded random feasible selection | 0 | cost-matched planner baseline |
| `cost_greedy` | marginal utility/cost selection | 0 | simple planner baseline |
| `all_feasible` | all compatible actions that fit | 0 | budget-use baseline |
| `oracle_selector` | exact small-set selection | 0 | diagnostic upper bound only; exclude from deployable-baseline claims |

The manuscript, reviews, and candidate evidence must be the same frozen inputs for every method. Planner comparisons must use the same candidate pool, budget, cost definitions, constraints, and downstream blind evaluator. `oracle_selector` uses estimated benefits and is not an observed-outcome oracle.

## Proposed method iterations

`methods/evidencepatch_risk_aware/` contains the original risk-aware proposed method used in the separate 100-paper M3 development run. `methods/evidencepatch_section_context/` is a new development variant that supplies BM25-retrieved complete manuscript sections to the risk estimator while keeping the emitted appendix limited to exact-source pointers. These proposed methods are intentionally not included in `METHODS`, which remains the fixed ten-control baseline registry. Their first-experiment runners import them directly and save separate run artifacts.

`methods/evidencepatch_bm25_risk_aware/` is the current leading implementation variant: it constructs candidates by local BM25 retrieval of exact source and review spans before M3 risk estimation. Its separate 100-paper generation and scoring run completed for 100/100 papers. The initial paired M3 proxy delta was +0.14/5 (95% CI [+0.03, +0.26]); two blind rescoring rounds on frozen outputs were +0.12 (CI [+0.02, +0.22]) and +0.11 (CI [−0.01, +0.22]). These are same-family exploratory score proxies, not independent generation replications or expert-confirmed resolution. See the result artifacts under `show/experiment/[1]first experiment/results/evidencepatch_bm25_riskaware_m3_100_v1/` and the two `..._repeat_02/` / `..._repeat_03/` folders.

## Context contract

Required for all methods: `manuscript: str`. Relevant optional fields: `paper_id`, `reviews` (string or structured value), `claims`, `evidence_candidates` (ordered list of `{claim, quote, response?}`), `candidates` (list of planner candidate dictionaries), `budget`, `seed`, `top_k`, `verification_feedback`, `planner_api`, and `llm_call`.

`llm_call` is a callable accepting keyword arguments `system`, `user`, `purpose`, and `paper_id`; it returns response text or a mapping with `content`. The caller owns API configuration, throttling, retries, usage logging, and credential handling.

## Upstream sources and reuse notes

Each method directory has a `SOURCE.md` recording the paper/repository link, its comparison role, and whether this implementation is a reproduction. The first-experiment registry includes both published-method-inspired adapters and transparent local controls; it does not imply that all ten entries are published algorithms.

The selected literature comparators are DRPG and Peerify. Their official repositories are source pointers only; the local adapters are independently written prompts, not upstream code reproductions. Upstream repository links, roles, and license status are recorded in `sources.json`. No upstream code is vendored because a reusable license has not been verified. The other eight registered conditions are explicitly local experimental controls, not claimed published algorithms.

These related papers address rebuttal generation, review verification, or actionable review generation. Their task settings do not all match this experiment's author-side, evidence-constrained appendix patching. The other entries are explicit local controls (generic generation, retrieval ablation, or planner policies), with no external repository claimed. This scope distinction is part of the benchmark definition.

## Runner integration

The first-experiment runners import methods through this registry or `baseline_method_adapter.py`; `run_method_benchmark_100.py` invokes the complete registered list. This package is scoped to that experiment. No GPU is used by these implementations. API-backed methods run only when the caller provides `llm_call`.

## Completed MiniMax-M3 panel

The 100-paper, ten-condition exploratory panel is stored at `show/experiment/[1]first experiment/results/baseline_methods_m3_100_v1/`. It includes 1,000 blind score cells, 1,335 logged MiniMax-M3 API attempts, and 14,812,401 reported tokens. Run `python3 'show/experiment/[1]first experiment/evaluation/validate_baseline_methods_m3_100_v1.py'` to audit paper/method coverage, output and score cells, source hashes, and API accounting. The panel is a single-judge score proxy; it does not establish expert-confirmed concern resolution. See the run's `statistical_report.md` and `failure_analysis.md` for results and limitations.

The joined 12-condition report at `show/experiment/[1]first experiment/results/all_methods_m3_100_audit/cost_score_tradeoff.md` adds both EvidencePatch variants, generation-token counts, primary score changes, and a unified repeat-03 score for each of the same 100 papers. It is a descriptive join across separate generation runs, not a cost-matched randomized effect: upstream candidate-pool costs and local CPU planner time are unmeasured. For example, section-context has primary/repeat-03 M3 changes of 0.00/+0.13; the repeat-03 Holm-adjusted p-value is 0.586, so this is not a supported method win. All twelve conditions are score proxies awaiting expert outcome labels.

The risk-aware run also has a deterministic candidate-coverage diagnostic at `show/experiment/[1]first experiment/results/evidencepatch_risk_aware_m3_100_v1/diagnostics/coverage_analysis.md`. It accounts for ineligible concern candidates and distinguishes exact-source provenance failures from requests explicitly labeled as requiring new evidence. The analysis uses frozen state, makes no API calls, and assigns no human labels.
