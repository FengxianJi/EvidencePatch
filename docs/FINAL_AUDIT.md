# Final public-resource audit — 2026-11-18

## Passed checks

- Public resource and blog working trees were reviewed independently of the private experiment workspace.
- Public code installed successfully into a new Python 3.12 environment. Imports and execution were checked from outside the source directory, including the method registry, BM25 retrieval, planning, verification, API-client import and statistical helpers. No API request or GPU was used in this installation audit.
- The numeric reproduction script recovers 300 papers, 284 eligible papers, 873 concern pairs; C5−C0 score mean −0.233333, eligible paper-macro proxy +0.055869, and Spearman rho 0.068744.
- Among proxy-improved papers, score direction counts remain 10 increased / 20 unchanged / 25 decreased.
- Offline retrieval and budget-control examples execute successfully. The frozen authored casebook retains outputs for ten controls and three proposed variants, 19 screenshot files, and four boundary situations.
- Public Markdown links were checked; two obsolete private-workspace links were replaced by public scope and result-table links.
- The final bilingual blog retains four display formulas, thirteen figure references and the confirmed author order. Its previous browser verification loaded all thirteen images, rendered math, and found no horizontal document overflow at 390px width.
- Public resource files were checked against the local credential material without printing it; private input files, request ledgers, caches, and training artifacts are excluded from Git and packaging.

## Scope limitations that remain explicit

Independent expert labels, original raw papers/reviews and full bootstrap/subgroup input assignments are not included. The public scripts do not rerun the original end-to-end experiment. Before/after screenshots use authored synthetic inputs; the two prose/anchor editing illustrations are hand-authored. These limitations should not be removed or replaced by efficacy claims.

No project-wide LICENSE file has been selected. The release is a public resource package with recorded license-file status, not an asserted MIT/Apache distribution.

## Archive verification

The packaging command reads clean Git tracked snapshots only, emits ZIP/tar.gz plus SHA256SUMS, and verifies ZIP CRC and every intended member byte. Bundle-specific commit IDs, inventories and checksums are emitted with each archive; they are not self-referential entries in this audit document.
