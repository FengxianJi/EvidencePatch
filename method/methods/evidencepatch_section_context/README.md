# EvidencePatch with retrieved section context

This is a development-only variant of the risk-aware EvidencePatch method. It supplies each candidate with the exact quotation plus BM25-ranked complete manuscript sections (top 5, at most 20,000 characters per candidate) before MiniMax-M3 estimates concern validity, quote relevance, uncertainty, and risk. The planner and final appendix constraints match the existing risk-aware arm; retrieved sections inform selection but are not copied into the appendix.

The method does not add experimental results or factual paraphrases. Selected appendix entries remain exact-source quotation pointers. The experiment uses the first 100 papers of the frozen Re² Main-120 development cohort and is separate from both the 300-paper confirmatory run and the original 100-paper baseline benchmark.

Run with:

```bash
python3 'show/experiment/[1]first experiment/run_evidencepatch_section_context_100.py'
```

Validate with:

```bash
python3 'show/experiment/[1]first experiment/evaluation/validate_evidencepatch_section_context_100.py'
```

The MiniMax-M3 score is a same-family exploratory proxy. It is not expert-confirmed concern resolution, and the Main-120 cohort is not cleared for redistribution. The runner uses no GPU and records per-paper input hashes, retrieved-section offsets and text hashes, API usage, and randomized score order locally. The validator produces appendix-output hashes from the saved state.

The 100-paper run is complete: mean paired score change was 0.000/5 (95% bootstrap CI [−0.12, +0.12]), with 18 increases, 64 ties, and 18 decreases. Independent integrity validation passed. See [`EXPLORATORY_RESULTS.md`](../../../show/experiment/%5B1%5Dfirst%20experiment/results/evidencepatch_section_context_m3_100_v1/EXPLORATORY_RESULTS.md) for the protocol, cost, and full result summary.
