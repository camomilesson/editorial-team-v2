# Retrieval evaluation report

## Design and dataset

Fixed corpus: 27 artifacts / 28 chunks. 
Cases: 12. Corpus SHA-256: `5b95072594c43aa662ffb3591745911467b5047b2523361f3056e624d86a9112`. 
Case SHA-256: `1511c4698022c530aa5655b5903b909e24d2a2c44330b25c53083481d634b3df`.

Metrics preserve the exact final SearchResult order. Precision uses requested k as the 
denominator. MRR is truncated at k. Empty-golden cases are N/A and excluded.

## Aggregate metrics

| Rerank | k | Hit rate | Precision | Recall | MRR@k | nDCG@k |
|---|---:|---:|---:|---:|---:|---:|
| off | 1 | 1.0000 | 1.0000 | 0.9545 | 1.0000 | 1.0000 |
| off | 3 | 1.0000 | 0.3636 | 1.0000 | 1.0000 | 1.0000 |
| off | 5 | 1.0000 | 0.2182 | 1.0000 | 1.0000 | 1.0000 |
| off | 10 | 1.0000 | 0.1091 | 1.0000 | 1.0000 | 1.0000 |
| on | 1 | 0.8182 | 0.8182 | 0.7727 | 0.8182 | 0.8182 |
| on | 3 | 1.0000 | 0.3636 | 1.0000 | 0.9091 | 0.9329 |
| on | 5 | 1.0000 | 0.2182 | 1.0000 | 0.9091 | 0.9329 |
| on | 10 | 1.0000 | 0.1091 | 1.0000 | 0.9091 | 0.9329 |

## Per-case qualitative stage analysis

### Reranking off

| Case | Dense relevant ranks | BM25 relevant ranks | RRF relevant ranks | Final relevant ranks |
|---|---|---|---|---|
| ret-001 | `chunk-v1-2743cefb0ced9c3d23e32eabf6e88f109e166def113b1bdaae4400c32e5bc574`: 1 | `chunk-v1-2743cefb0ced9c3d23e32eabf6e88f109e166def113b1bdaae4400c32e5bc574`: 1 | `chunk-v1-2743cefb0ced9c3d23e32eabf6e88f109e166def113b1bdaae4400c32e5bc574`: 1 | `chunk-v1-2743cefb0ced9c3d23e32eabf6e88f109e166def113b1bdaae4400c32e5bc574`: 1; Both branches found the evidence; fusion preserved it without a unique fix. |
| ret-002 | `chunk-v1-0f7a53d3ac9ade134b608f743c44f75aeb435b0c962063464e6a8639946ad799`: 1 | `chunk-v1-0f7a53d3ac9ade134b608f743c44f75aeb435b0c962063464e6a8639946ad799`: 1 | `chunk-v1-0f7a53d3ac9ade134b608f743c44f75aeb435b0c962063464e6a8639946ad799`: 1 | `chunk-v1-0f7a53d3ac9ade134b608f743c44f75aeb435b0c962063464e6a8639946ad799`: 1; Both branches found the evidence; fusion preserved it without a unique fix. |
| ret-003 | `chunk-v1-cb8453ee7b28ec20b3f97e4e505c648d2e3c8dfb3886e89d56214cdf8558f309`: 1 | `chunk-v1-cb8453ee7b28ec20b3f97e4e505c648d2e3c8dfb3886e89d56214cdf8558f309`: 1 | `chunk-v1-cb8453ee7b28ec20b3f97e4e505c648d2e3c8dfb3886e89d56214cdf8558f309`: 1 | `chunk-v1-cb8453ee7b28ec20b3f97e4e505c648d2e3c8dfb3886e89d56214cdf8558f309`: 1; Both branches found the evidence; fusion preserved it without a unique fix. |
| ret-004 | `chunk-v1-dc00b85708178f578b892dadadaf44b6fb1bd0d602e4bc32ecef52683a137505`: 1 | `chunk-v1-dc00b85708178f578b892dadadaf44b6fb1bd0d602e4bc32ecef52683a137505`: 1 | `chunk-v1-dc00b85708178f578b892dadadaf44b6fb1bd0d602e4bc32ecef52683a137505`: 1 | `chunk-v1-dc00b85708178f578b892dadadaf44b6fb1bd0d602e4bc32ecef52683a137505`: 1; Both branches found the evidence; fusion preserved it without a unique fix. |
| ret-005 | `chunk-v1-0a97a57f52f85cb54123b6f634fa3e1c80c83c2661b80a9d98242099bfd25271`: 2 | `chunk-v1-0a97a57f52f85cb54123b6f634fa3e1c80c83c2661b80a9d98242099bfd25271`: 1 | `chunk-v1-0a97a57f52f85cb54123b6f634fa3e1c80c83c2661b80a9d98242099bfd25271`: 1 | `chunk-v1-0a97a57f52f85cb54123b6f634fa3e1c80c83c2661b80a9d98242099bfd25271`: 1; BM25 ranked the best relevant evidence above dense search. |
| ret-006 | `chunk-v1-306b1b6c0b8bd82b13368f02adcce81bc816dfc2954d59578cc537294247ca72`: 1 | `chunk-v1-306b1b6c0b8bd82b13368f02adcce81bc816dfc2954d59578cc537294247ca72`: 1 | `chunk-v1-306b1b6c0b8bd82b13368f02adcce81bc816dfc2954d59578cc537294247ca72`: 1 | `chunk-v1-306b1b6c0b8bd82b13368f02adcce81bc816dfc2954d59578cc537294247ca72`: 1; Both branches found the evidence; fusion preserved it without a unique fix. |
| ret-007 | `chunk-v1-2c4b65309b03101aa92dd11edeb99ba2222e74a612743bfb349b4e4927125008`: 2 | `chunk-v1-2c4b65309b03101aa92dd11edeb99ba2222e74a612743bfb349b4e4927125008`: 1 | `chunk-v1-2c4b65309b03101aa92dd11edeb99ba2222e74a612743bfb349b4e4927125008`: 1 | `chunk-v1-2c4b65309b03101aa92dd11edeb99ba2222e74a612743bfb349b4e4927125008`: 1; BM25 ranked the best relevant evidence above dense search. |
| ret-008 | `chunk-v1-91f30d9ee2ae93767ea549a7196c085c8e47185718d9592c919b604d182a72a6`: 1 | `chunk-v1-91f30d9ee2ae93767ea549a7196c085c8e47185718d9592c919b604d182a72a6`: 1 | `chunk-v1-91f30d9ee2ae93767ea549a7196c085c8e47185718d9592c919b604d182a72a6`: 1 | `chunk-v1-91f30d9ee2ae93767ea549a7196c085c8e47185718d9592c919b604d182a72a6`: 1; Both branches found the evidence; fusion preserved it without a unique fix. |
| ret-009 | `chunk-v1-cb90a8d6c3c0cb7ccd22da8a753785be73b48cde8e0fe803e0ea0daa79665c4a`: 1 | `chunk-v1-cb90a8d6c3c0cb7ccd22da8a753785be73b48cde8e0fe803e0ea0daa79665c4a`: 2 | `chunk-v1-cb90a8d6c3c0cb7ccd22da8a753785be73b48cde8e0fe803e0ea0daa79665c4a`: 1 | `chunk-v1-cb90a8d6c3c0cb7ccd22da8a753785be73b48cde8e0fe803e0ea0daa79665c4a`: 1; Dense ranked the best relevant evidence above BM25. |
| ret-010 | `chunk-v1-91f30d9ee2ae93767ea549a7196c085c8e47185718d9592c919b604d182a72a6`: 1 | `chunk-v1-91f30d9ee2ae93767ea549a7196c085c8e47185718d9592c919b604d182a72a6`: 1 | `chunk-v1-91f30d9ee2ae93767ea549a7196c085c8e47185718d9592c919b604d182a72a6`: 1 | `chunk-v1-91f30d9ee2ae93767ea549a7196c085c8e47185718d9592c919b604d182a72a6`: 1; Both branches found the evidence; fusion preserved it without a unique fix. |
| ret-011 | `chunk-v1-89f0d47369ae232a3073868195fd229a4c1908284ea6aa4be099b5bd2a983218`: 1, `chunk-v1-cb8453ee7b28ec20b3f97e4e505c648d2e3c8dfb3886e89d56214cdf8558f309`: 2 | `chunk-v1-89f0d47369ae232a3073868195fd229a4c1908284ea6aa4be099b5bd2a983218`: 1, `chunk-v1-cb8453ee7b28ec20b3f97e4e505c648d2e3c8dfb3886e89d56214cdf8558f309`: 2 | `chunk-v1-89f0d47369ae232a3073868195fd229a4c1908284ea6aa4be099b5bd2a983218`: 1, `chunk-v1-cb8453ee7b28ec20b3f97e4e505c648d2e3c8dfb3886e89d56214cdf8558f309`: 2 | `chunk-v1-89f0d47369ae232a3073868195fd229a4c1908284ea6aa4be099b5bd2a983218`: 1, `chunk-v1-cb8453ee7b28ec20b3f97e4e505c648d2e3c8dfb3886e89d56214cdf8558f309`: 2; Both branches found the evidence; fusion preserved it without a unique fix. |
| ret-012 | N/A | N/A | N/A | N/A; Out-of-corpus qualitative probe; deterministic metrics are N/A. |

### Reranking on

| Case | Dense relevant ranks | BM25 relevant ranks | RRF relevant ranks | Final relevant ranks |
|---|---|---|---|---|
| ret-001 | `chunk-v1-2743cefb0ced9c3d23e32eabf6e88f109e166def113b1bdaae4400c32e5bc574`: 1 | `chunk-v1-2743cefb0ced9c3d23e32eabf6e88f109e166def113b1bdaae4400c32e5bc574`: 1 | `chunk-v1-2743cefb0ced9c3d23e32eabf6e88f109e166def113b1bdaae4400c32e5bc574`: 1 | `chunk-v1-2743cefb0ced9c3d23e32eabf6e88f109e166def113b1bdaae4400c32e5bc574`: 1; Both branches found the evidence; fusion preserved it without a unique fix. |
| ret-002 | `chunk-v1-0f7a53d3ac9ade134b608f743c44f75aeb435b0c962063464e6a8639946ad799`: 1 | `chunk-v1-0f7a53d3ac9ade134b608f743c44f75aeb435b0c962063464e6a8639946ad799`: 1 | `chunk-v1-0f7a53d3ac9ade134b608f743c44f75aeb435b0c962063464e6a8639946ad799`: 1 | `chunk-v1-0f7a53d3ac9ade134b608f743c44f75aeb435b0c962063464e6a8639946ad799`: 1; Both branches found the evidence; fusion preserved it without a unique fix. |
| ret-003 | `chunk-v1-cb8453ee7b28ec20b3f97e4e505c648d2e3c8dfb3886e89d56214cdf8558f309`: 1 | `chunk-v1-cb8453ee7b28ec20b3f97e4e505c648d2e3c8dfb3886e89d56214cdf8558f309`: 1 | `chunk-v1-cb8453ee7b28ec20b3f97e4e505c648d2e3c8dfb3886e89d56214cdf8558f309`: 1 | `chunk-v1-cb8453ee7b28ec20b3f97e4e505c648d2e3c8dfb3886e89d56214cdf8558f309`: 1; Both branches found the evidence; fusion preserved it without a unique fix. |
| ret-004 | `chunk-v1-dc00b85708178f578b892dadadaf44b6fb1bd0d602e4bc32ecef52683a137505`: 1 | `chunk-v1-dc00b85708178f578b892dadadaf44b6fb1bd0d602e4bc32ecef52683a137505`: 1 | `chunk-v1-dc00b85708178f578b892dadadaf44b6fb1bd0d602e4bc32ecef52683a137505`: 1 | `chunk-v1-dc00b85708178f578b892dadadaf44b6fb1bd0d602e4bc32ecef52683a137505`: 1; Both branches found the evidence; fusion preserved it without a unique fix. |
| ret-005 | `chunk-v1-0a97a57f52f85cb54123b6f634fa3e1c80c83c2661b80a9d98242099bfd25271`: 2 | `chunk-v1-0a97a57f52f85cb54123b6f634fa3e1c80c83c2661b80a9d98242099bfd25271`: 1 | `chunk-v1-0a97a57f52f85cb54123b6f634fa3e1c80c83c2661b80a9d98242099bfd25271`: 1 | `chunk-v1-0a97a57f52f85cb54123b6f634fa3e1c80c83c2661b80a9d98242099bfd25271`: 2; BM25 ranked the best relevant evidence above dense search. Reranking demoted the first relevant chunk. |
| ret-006 | `chunk-v1-306b1b6c0b8bd82b13368f02adcce81bc816dfc2954d59578cc537294247ca72`: 1 | `chunk-v1-306b1b6c0b8bd82b13368f02adcce81bc816dfc2954d59578cc537294247ca72`: 1 | `chunk-v1-306b1b6c0b8bd82b13368f02adcce81bc816dfc2954d59578cc537294247ca72`: 1 | `chunk-v1-306b1b6c0b8bd82b13368f02adcce81bc816dfc2954d59578cc537294247ca72`: 1; Both branches found the evidence; fusion preserved it without a unique fix. |
| ret-007 | `chunk-v1-2c4b65309b03101aa92dd11edeb99ba2222e74a612743bfb349b4e4927125008`: 2 | `chunk-v1-2c4b65309b03101aa92dd11edeb99ba2222e74a612743bfb349b4e4927125008`: 1 | `chunk-v1-2c4b65309b03101aa92dd11edeb99ba2222e74a612743bfb349b4e4927125008`: 1 | `chunk-v1-2c4b65309b03101aa92dd11edeb99ba2222e74a612743bfb349b4e4927125008`: 1; BM25 ranked the best relevant evidence above dense search. |
| ret-008 | `chunk-v1-91f30d9ee2ae93767ea549a7196c085c8e47185718d9592c919b604d182a72a6`: 1 | `chunk-v1-91f30d9ee2ae93767ea549a7196c085c8e47185718d9592c919b604d182a72a6`: 1 | `chunk-v1-91f30d9ee2ae93767ea549a7196c085c8e47185718d9592c919b604d182a72a6`: 1 | `chunk-v1-91f30d9ee2ae93767ea549a7196c085c8e47185718d9592c919b604d182a72a6`: 1; Both branches found the evidence; fusion preserved it without a unique fix. |
| ret-009 | `chunk-v1-cb90a8d6c3c0cb7ccd22da8a753785be73b48cde8e0fe803e0ea0daa79665c4a`: 1 | `chunk-v1-cb90a8d6c3c0cb7ccd22da8a753785be73b48cde8e0fe803e0ea0daa79665c4a`: 2 | `chunk-v1-cb90a8d6c3c0cb7ccd22da8a753785be73b48cde8e0fe803e0ea0daa79665c4a`: 1 | `chunk-v1-cb90a8d6c3c0cb7ccd22da8a753785be73b48cde8e0fe803e0ea0daa79665c4a`: 1; Dense ranked the best relevant evidence above BM25. |
| ret-010 | `chunk-v1-91f30d9ee2ae93767ea549a7196c085c8e47185718d9592c919b604d182a72a6`: 1 | `chunk-v1-91f30d9ee2ae93767ea549a7196c085c8e47185718d9592c919b604d182a72a6`: 1 | `chunk-v1-91f30d9ee2ae93767ea549a7196c085c8e47185718d9592c919b604d182a72a6`: 1 | `chunk-v1-91f30d9ee2ae93767ea549a7196c085c8e47185718d9592c919b604d182a72a6`: 2; Both branches found the evidence; fusion preserved it without a unique fix. Reranking demoted the first relevant chunk. |
| ret-011 | `chunk-v1-89f0d47369ae232a3073868195fd229a4c1908284ea6aa4be099b5bd2a983218`: 1, `chunk-v1-cb8453ee7b28ec20b3f97e4e505c648d2e3c8dfb3886e89d56214cdf8558f309`: 2 | `chunk-v1-89f0d47369ae232a3073868195fd229a4c1908284ea6aa4be099b5bd2a983218`: 1, `chunk-v1-cb8453ee7b28ec20b3f97e4e505c648d2e3c8dfb3886e89d56214cdf8558f309`: 2 | `chunk-v1-89f0d47369ae232a3073868195fd229a4c1908284ea6aa4be099b5bd2a983218`: 1, `chunk-v1-cb8453ee7b28ec20b3f97e4e505c648d2e3c8dfb3886e89d56214cdf8558f309`: 2 | `chunk-v1-89f0d47369ae232a3073868195fd229a4c1908284ea6aa4be099b5bd2a983218`: 1, `chunk-v1-cb8453ee7b28ec20b3f97e4e505c648d2e3c8dfb3886e89d56214cdf8558f309`: 2; Both branches found the evidence; fusion preserved it without a unique fix. |
| ret-012 | N/A | N/A | N/A | N/A; Out-of-corpus qualitative probe; deterministic metrics are N/A. |

## Reranking outcome by case

| Case | Outcome at maximum k |
|---|---|
| ret-001 | unchanged |
| ret-002 | unchanged |
| ret-003 | unchanged |
| ret-004 | unchanged |
| ret-005 | worsened |
| ret-006 | unchanged |
| ret-007 | unchanged |
| ret-008 | unchanged |
| ret-009 | unchanged |
| ret-010 | worsened |
| ret-011 | unchanged |
| ret-012 | not_applicable |

## Interpretation

Compare dense and BM25 ranks to identify semantic and exact-term recovery; compare 
RRF and final ranks to identify fusion and reranking changes. Neutral and negative 
reranking 
deltas are retained in the JSON rather than hidden.

As k increases, recall and hit rate can rise while requested-k precision generally 
falls because every additional slot remains in the denominator.

Empty-golden cases: 1.
They are qualitative out-of-corpus probes only; no abstention threshold is 
introduced.

## Limitations

Binary relevance and a fixed corpus cannot measure generation quality, agent 
behavior, 
or production-distribution drift. This milestone makes no such claims.
