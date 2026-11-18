# Method provenance

- Upstream paper/code: none claimed.
- Implementation: local BM25 retrieval over frozen Markdown sections/passages, deterministic source/review span extraction, MiniMax-M3 candidate risk estimation, and the project budget planner.
- Reproduction status: new development variant, not an upstream reproduction.
- All selected review/source strings are checked as exact normalized substrings before scoring.
