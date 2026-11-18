# EvidencePatch with deterministic BM25 candidates

Development variant of `evidencepatch_risk_aware`. It retrieves exact manuscript passages locally for each human review, pairs them with exact excerpts from the corresponding review, asks MiniMax-M3 to estimate concern validity/relevance/risk, and applies the same budget planner. It never creates new experimental results or factual paraphrases.

The candidate pool differs from the frozen v2 locator used in the other two 100-paper EvidencePatch runs. Treat results as a separate exploratory generation run, not a randomized comparison against those variants. It is not an upstream paper-code reproduction and has no external repository claimed.
