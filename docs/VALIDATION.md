# Validation plan

This report is a reproducible engineering validation plan, not a clinical study.

## Required baselines
1. deterministic template runtime
2. safety + memory
3. safety + memory + latent state
4. full constrained runtime

## Required ablations
- remove memory
- remove state
- remove adaptive policy
- remove personalization
- remove multimodal inputs
- remove response verifier

## Required measurements
Safety recall/FNR/FPR, memory precision@k, state calibration, policy constraint violations, response verifier rejection rate, latency p50/p95/p99, encrypted-storage round trip, and memory footprint.

No numerical SOTA or clinical superiority claim should be made until an independently governed dataset and evaluation protocol are available.
