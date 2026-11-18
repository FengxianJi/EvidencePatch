# Public data and result inventory

## Anonymous numeric endpoints

`results/numeric/paper_endpoints.csv`: 300 shuffled rows with no original IDs, titles, source hashes or text.

- `baseline_score`: M3 mean over three ratings of C0.
- `score_delta`: C5 mean score minus C0 mean score.
- `score_delta_vs_placebo`: C5 mean score minus C1 mean score.
- `resolution_delta`: paper-macro difference of concern proxy values (unresolved=0, partial=0.5, full=1), conditional on model-labeled valid/repairable concerns. Empty for 16 papers with no eligible denominator.
- `eligible_concerns`: count of concerns in the paper's proxy denominator.

There are five columns: four score/proxy endpoints or differences plus one count. Paper rows are descriptive model outputs, not expert labels.

`results/numeric/concern_transitions.csv`: counts grouped by actionability and C0/C5 label transition; total 873 eligible pairs. No per-concern text or original identifiers are released.

## Aggregate tables

- `primary300_contrasts.csv`: five-condition content intervention, three M3 ratings per condition.
- `primary300_model_family.csv`: evaluator-family estimates and same-cohort rerun.
- `baseline_methods_100.csv`: ten baseline/control conditions on 100 development papers.
- `bm25_riskaware_repeatability.csv`: initial BM25 run and frozen-output scoring repeats.
- `all_methods_repeat03.csv`: uniform scoring of 12 frozen method/control outputs on the development cohort.

Where tables retain `source_metrics`, those strings refer to the historical local run, not an included raw artifact. Hashes pin the archived aggregate source. Result panels should not be pooled as independent studies or called one contemporaneous cost-matched comparison.

The subgroup archive covers overlapping, post-hoc groups. Public paper endpoints support baseline-score stratification but deliberately do not expose per-paper concern-category assignments; category-specific group summaries are archived aggregates, not fully reconstructible from these numeric records.

## What can be reproduced

Public scripts reproduce the headline point estimates and chart layouts without paid model calls. Original API runs, quote provenance judgments, full subgroup assignment, and end-to-end manuscript experiments need the original inputs and provider access, which are not bundled. Expert labels are absent. All source-derived text remains outside this release.
