# Release scope and reproducibility coverage

This is the blog's **public code and results resource release**. It is not a full raw-manuscript dataset or a claim of complete end-to-end study reproduction.

| Material | Included | Publicly reproducible scope |
|---|---|---|
| Ten registered controls and three EvidencePatch variants | Yes | Code inspection and supplied-input execution |
| Parsing, retrieval, budget planning, execution and verification modules | Yes | Reusable implemented components; PDF parsers optional |
| Headline primary score and concern-proxy endpoints | Yes | Recompute point estimates from anonymous numeric records |
| Main/baseline/rescoring tables and figure exports | Yes | Inspect all reported aggregate tables; feedback plots can be rebuilt |
| Original bootstrap intervals and subgroup analyses | Archived summaries | Not all original repeated-score and subgroup assignments are public |
| Thirteen-method before/after casebook | Yes | Render frozen synthetic outputs and screenshots without API calls |
| Model regeneration of the casebook | Input + driver included | Requires provider access; model outputs can vary |
| Original source papers, reviews, IDs and raw model responses | No | Not available from this public release |
| Expert-adjudicated resolution labels | No | Not collected; no expert efficacy claim |
| Full-body rewrite efficacy and monotonic multiround gains | No | Not established by these experiments |
| Incomplete Track B training panel | No final outcome | Auxiliary unfinished work is not packaged as a completed result |

The 300-paper experiment evaluates frozen-body content conditions. The 100-paper panels compare method/control outputs on a separate development cohort. The authored casebook demonstrates mechanism and editing surfaces; it supplies no scientific measurements or study review scores. These three sources of evidence must not be pooled.

Historical `protocol_sources/` and method documentation retain context from the private workspace. Supported public commands are listed in the repository README. Published checksum manifests pin the exported artifacts after adaptation.

No repository-wide `LICENSE` or `COPYING` file has been assigned in this release. Do not describe the package as MIT- or Apache-licensed. Third-party paper and repository references do not assign a license to those third-party materials. This document records release metadata, not a new grant of rights.
