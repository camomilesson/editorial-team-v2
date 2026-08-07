# Fixed retrieval evaluation

This isolated HW2 evaluation seeds a dedicated SQLite database with the fixed corpus, resolves
content anchors through the production `ParagraphChunker`, and evaluates the exact ordered stage
rankings from `HybridRetriever.search_with_stages`.

Run from the repository root:

```bash
python evaluation/retrieval/run_retrieval_eval.py
```

Optional flags are `--database`, `--output`, `--report`, `--rerank both|on|off`, and
`--k 1 3 5 10`. Without `--database`, a temporary database is used and removed. The default
reports under `evaluation/outputs/` are reproducible submission artifacts; model caches and
SQLite databases remain ignored.

Precision divides by requested k even when fewer results are returned. MRR is truncated at k.
Duplicate predicted chunk IDs are rejected. Empty-golden cases are reported as N/A and excluded
from all five aggregate metrics.
