# Running the implementation

The package API is `method.get_method(method_id)(context)`. Ten controls are registered in `method.registry.METHODS`; proposed variants are imported directly from `method.methods.evidencepatch_*`. All return at least `method_id` and `appendix_markdown`.

Context fields include `manuscript`, `reviews`, `evidence_candidates`, `candidates`, `budget`, `seed`, `top_k`, `planner_api`, and optional `llm_call`. The CLI supplies `planner_api`; API mode also supplies `llm_call`.

Candidate fields are `candidate_id`, `claim`, `quote`, `cost`, `expected_benefit`, `uncertainty`, `risk`; variants can use `review_context`, `section`, `bm25_score`, and `manuscript_section_context`. See each method's source for its contract. The BM25 risk-aware adapter expects preconstructed exact-span pairs. The retrieval utilities in `diagnosis.evidence_retrieval` construct BM25 passage candidates; the toy CLI does not automatically reconstruct the original candidate-pool experiment.

Offline examples:

```bash
python3 scripts/run_method.py --method frozen_original
python3 scripts/run_method.py --method retrieval_only
python3 scripts/run_method.py --method random_budget
python3 scripts/run_method.py --method cost_greedy
```

The example's expected benefits are illustrative inputs, not model estimates or measured research outcomes. For an actual study, freeze the candidate pool, define budgets/costs, and separate candidate selection from evaluation.

Original internal method documents include historical paths and results links. The repository README and public CLI define the release's supported entry points. Manuscript ingestion can optionally use PyMuPDF or `mutool` for PDFs; Markdown parsing and BM25 need only the Python standard library. No optional parser package is required for the shipped examples.
