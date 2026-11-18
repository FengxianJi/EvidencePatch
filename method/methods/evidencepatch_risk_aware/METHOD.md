# EvidencePatch risk-aware selector

## Role

This is the proposed method arm for an exploratory comparison against the ten registered baselines. It is implemented as a separate adapter so the completed ten-method benchmark and its recorded source hashes remain unchanged.

## Input and action space

The method receives the frozen first-experiment manuscript, review excerpts, and the exact-source-verified candidate pool already shared by the planner baselines. Each candidate pairs one review concern with a verbatim manuscript quotation. The method cannot add a new experiment, result, citation, or factual claim.

## Decision rule

MiniMax-M3 estimates, per candidate, concern validity, quote relevance, uncertainty, and risk. For candidate (i), expected benefit is the product of the first two probabilities. The budget planner maximizes marginal risk-adjusted utility per cost:

`utility_i = p_valid_i * p_relevant_i - uncertainty_i - risk_i`

Overlapping coverage is represented with claim-level benefits, so a concern is not counted more than once. The budget and candidate pool match the ten-method experiment. Missing or malformed estimates abstain. Only selected exact-source quotations are rendered, alongside a fixed statement that they are locators rather than proof of resolution.

## Evaluation status and limits

The paired evaluation uses a fresh frozen-original score and a method score, with randomized blind order, on the same 100 Re² papers. The scoring model is also the estimator model, so this is not independent validation. There is one score draw per condition; decoding defaults are not pinned. The run is exploratory and cannot establish expert-confirmed resolution or performance on the separate primary-300 cohort.
