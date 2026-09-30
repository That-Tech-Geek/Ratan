# Semantic memory

Ratan uses a deterministic embedding/index implementation as the portable baseline. It is intentionally not described as HNSW until a production ANN backend is benchmarked. `SemanticMemory` exposes the stable insert/retrieve contract; a future HNSW implementation can satisfy the same contract.

Memory scores combine semantic similarity, importance, temporal relevance and a sensitivity penalty.
