# Judged generation evaluation

This standalone RAG harness reuses the fixed retrieval corpus and production retriever. It does
not invoke Coordinator, tools, Telegram, or Writer–Critic–Editor. The exact retrieval order is
passed to a grounded generator, then four separate strict structured Gemini judge calls score
faithfulness, answer relevance, context precision, and context recall.

The course repository did not contain the exact Session 11 §5 category names, so cases use the
transparent requested mapping: missing relevant context, irrelevant/distracting context,
incomplete multi-chunk context, unsupported claim/hallucination, near-duplicate/conflicting
context, and out-of-corpus/required abstention.

Run `python evaluation/generation/run_generation_eval.py`. Configure optional
`EDITORIAL_EVAL_GENERATOR_MODEL`, `EDITORIAL_EVAL_JUDGE_MODEL`, and
`EDITORIAL_EVAL_CACHE_PATH`; models fall back to `AGENT_MODEL`. The cache is content-addressed,
contains no credentials, and is ignored by Git.
