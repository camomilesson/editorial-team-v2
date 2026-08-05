# Judged generation evaluation

Cases: 20 rerank-on; 9 stratified rerank-off. 
Generator: `gemini-3.1-flash-lite`. Judge: `gemini-3.1-flash-lite`.

## Failure-category distribution

- `incomplete_multi_chunk_context`: 3
- `irrelevant_distracting_context`: 4
- `missing_relevant_context`: 4
- `near_duplicate_conflicting_context`: 4
- `out_of_corpus_required_abstention`: 3
- `unsupported_claim_hallucination`: 2

## Aggregate metrics

| Condition | Faithfulness | Answer relevance | Context precision | Context recall |
|---|---:|---:|---:|---:|
| rerank on | 0.9000 | 0.8450 | 0.2225 | 0.9000 |
| rerank on comparison subset | 0.8889 | 0.8222 | 0.2833 | 0.8889 |
| rerank off subset | 0.8889 | 0.8667 | 0.2889 | 0.8667 |

## Category-level metrics (rerank on)

| Category | Faithfulness | Answer relevance | Context precision | Context recall |
|---|---:|---:|---:|---:|
| incomplete_multi_chunk_context | 0.3333 | 0.2000 | 0.1333 | 0.3333 |
| irrelevant_distracting_context | 1.0000 | 1.0000 | 0.2875 | 1.0000 |
| missing_relevant_context | 1.0000 | 1.0000 | 0.2125 | 1.0000 |
| near_duplicate_conflicting_context | 1.0000 | 0.9000 | 0.4125 | 1.0000 |
| out_of_corpus_required_abstention | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| unsupported_claim_hallucination | 1.0000 | 0.8500 | 0.2000 | 1.0000 |

## Reranking comparison

Improved: gen-003, gen-009, gen-012.  
Unchanged: gen-001, gen-007, gen-011, gen-017.  
Worsened: gen-004, gen-020.

## Out-of-corpus behavior and retrieval/generation disagreements

gen-017: The corpus does not provide the answer. gen-018: The corpus does not provide the answer. gen-019: The corpus does not provide the answer.
Retrieval rank quality and answer quality are separate: correct context can still 
yield an 
unsupported or incomplete answer, while a rank change may leave the generated answer 
materially unchanged. Per-case answers, reasons, and chunk orders are retained in JSON.

The three out-of-corpus cases all abstained correctly. Their faithfulness, answer relevance, and
context recall were 1.0, while context precision was 0.0 because forced top-five retrieval still
returned unrelated chunks.

The strongest disagreement appeared in the incomplete multi-chunk category. Its rerank-on means
were 0.3333 faithfulness, 0.2000 answer relevance, 0.1333 context precision, and 0.3333 context
recall. `gen-011` and `gen-013` abstained despite receiving their golden chunk, while `gen-012`
answered from relevant material but did not make the requested comparison completely. Correct
retrieval therefore did not guarantee correct generation.

## Judge bias and limitations

LLM scores are not deterministic. Risks include position bias, verbosity preference, 
same-family self-preference, judge-model mismatch, and sensitivity to golden wording. 
Mitigations are fixed metric-specific rubrics, stable context order, hidden reranking 
condition, structured scores, prompt/model version recording, persistent caching, and 
category-level reporting. A fixed manual sample should still be inspected. This harness 
does not evaluate agents, routing, tools, or generation outside this standalone RAG path.

Cache: 0 hits / 116 misses.
