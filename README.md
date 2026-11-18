# EvidencePatch — code and experimental resources

Resources accompanying **EvidencePatch: When Does Revising a Paper from AI Reviews Actually Help?**

**[Read the blog (English)](https://fengxianji.github.io/FXJ-BLOG/en/research/2026/11/17/evidence-only-appendices/)** · **[中文博客](https://fengxianji.github.io/FXJ-BLOG/research/2026/11/17/evidence-only-appendices/)** · [Results tables](results/tables/) · [Figures](results/figures/) · [Anonymous numeric records](results/numeric/) · [Download repository ZIP](https://github.com/FengxianJi/EvidencePatch/archive/refs/heads/main.zip)

EvidencePatch is an evidence-grounded, budget-aware method family for responding to scientific review concerns. This release includes the implemented methods and controls, reusable parsing/retrieval/planning/verification modules, and the numeric results supporting the blog.

## What the results establish

The 300-paper frozen-body appendix study has a negative M3 recommendation-score contrast (C5−C0 = −0.233/5). In a separate same-family concern audit, the paper-macro resolution proxy increased by +0.0559 among 284 eligible papers and 873 concern pairs. Of 55 proxy-improved papers, 10 increased in recommendation score, 20 tied, and 25 decreased. These are different endpoints and denominators. Expert-confirmed resolution and scientific quality improvements have not been established.

The complete planner is implemented but is not validated by the C0–C5 content intervention alone. The 100-paper method panels are exploratory; frozen-output rescoring is not regeneration of the method. DRPG-style and Peerify-inspired adapters are locally written comparators, not faithful upstream reproductions.

## Quick start — no API or GPU required

Python 3.10+; no Python dependencies for the default commands:

```bash
git clone https://github.com/FengxianJi/EvidencePatch.git
cd EvidencePatch
python3 scripts/verify_release.py
python3 scripts/reproduce_results.py
python3 scripts/run_method.py
```

`reproduce_results.py` recomputes the primary mean score delta, eligible proxy mean, correlation, and direction counts from the anonymous numeric records. It writes `reproduced/summary.json`. No paper text is needed. The shipped confidence intervals are archived results; this command does not rerun the original model calls or bootstrap protocols.

`run_method.py` defaults to the retrieval-only baseline on an explicitly synthetic example. That example illustrates the interface and is not a study result. To try the budget control:

```bash
python3 scripts/run_method.py --method cost_greedy
```

## Directory guide

| Location | Contents |
|---|---|
| `method/` | Ten registered control conditions and three EvidencePatch implementation variants |
| `dataset/` | Corpus helpers, local parsing and version/hash utilities; no manuscripts |
| `diagnosis/` | Claim decomposition prompts and dependency-free BM25 retrieval |
| `planner/` | Budget-constrained selection, dependencies, conflicts and resource planning |
| `executor/` | MiniMax China-region client, retry logic and token accounting |
| `verifier/` | Lexical provenance and planning checks; lexical checks do not prove semantic support |
| `evaluation/` | Original reusable statistics functions |
| `scripts/` | Runnable public result reproduction, method CLI and figure generation |
| `results/numeric/` | Shuffled anonymous paper endpoints and aggregate concern transitions |
| `results/tables/` | Complete aggregate main, baseline and repeat-scoring tables |
| `results/analyses/` | Archived alignment and exploratory subgroup summaries |
| `results/figures/` | Main conceptual figure and experiment plots, with vector/raster exports |
| `protocol_sources/` | Historical experiment and analysis source snapshots for inspection |
| `docs/` | Data scope, protocol notes and source/export hashes |

## Rebuild the figures

Install `gnuplot` through your normal system package manager, then run:

```bash
python3 scripts/build_feedback_figures.py
```

This rebuilds the three feedback charts (SVG, PNG, PDF) and the main framework SVG from `results/figures/feedback/aggregate_data.json`. A main PNG is also provided; its ImageMagick export command is documented in the figure directory. Baseline benchmark figures are archived outputs, with their numeric tables shipped separately. Run checksum verification before rebuilding figures: regenerated PDF metadata may differ from the archived release bytes even when numeric results are unchanged.

## Use your own manuscripts and API credentials

All methods use a context dictionary. See `examples/toy_context.json` and `docs/RUNNING_METHODS.md`. API-backed methods require an explicit `--allow-api` flag:

```bash
# Copy the example config somewhere private and set its key_file to a private key file.
python3 scripts/run_method.py --method evidencepatch_bm25_risk_aware \
  --input /path/to/your/context.json \
  --api-config /path/to/private/config.json --allow-api
```

Calls can incur provider charges and transmit the supplied material to the configured service. The CLI does not launch the original 300-paper study. You supply candidate construction, input rights and your own credentials; no original raw corpus is bundled. `local_runs/` is ignored by Git.

## Scope of the public data

The release contains no source manuscripts, original reviews, quotations from them, original per-paper IDs, checkpoints, credentials or raw API responses. Anonymous numeric rows are shuffled and contain only five numeric endpoint/count columns. Their missing proxy entries correspond to 16 papers without an eligible concern denominator; missing values must not be treated as zero. Scope and schemas are in [docs/DATA.md](docs/DATA.md).

Archived protocol sources retain their historical workspace-relative assumptions and cannot reproduce the private original experiment without its inputs. They are provided for methodological inspection. The supported public commands are the four scripts listed above. [Source snapshot hashes](docs/source_snapshot.json) identify the exported originals; [release checksums](RELEASE_MANIFEST.json) identify the public files after adaptation.

The publication uses the same model family for concern labeling and its local-resolution judgment. Some supporting quotes failed exact match and two exceeded the word limit. Exploratory subgroup intervals are unadjusted; they cannot establish a reliable subgroup benefit. See [results/analyses/](results/analyses/) for the saved limitations.
