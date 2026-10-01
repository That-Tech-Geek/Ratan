# Attune

Privacy-first, on-device mental wellness companion built around a deterministic safety boundary, deterministic belief state, bounded contextual bandit policy, and clinician-authored template expression layer.

> Engineering preview. This repository is an implementation scaffold and research harness, not a clinical device and not a substitute for professional care.

## MVP: try it before spending GPU hours

The repository now includes a zero-dependency browser research preview in demo/index.html.

Run:

    python -m http.server 8000 --directory demo

Then open http://localhost:8000.

The MVP is intentionally lightweight. It is designed to answer product questions first: does the interaction feel useful, is the safety UX understandable, are the visible signals helpful, and what should be trained next?

It does not claim trained-model performance. No GPU, model weights, backend, or network connection is required.

## Architecture

mobile -> runtime -> safety -> belief -> policy -> expression

- Layer 0: hard pre/post safety shield. Vetoed inputs/actions never reach the policy selector.
- Layer 1: deterministic belief state with bounded valence/arousal updates, alliance, readiness, risk, and session momentum.
- Layer 2: bounded adaptive policy with explicit utility penalties and feasibility masks.
- Layer 3: deterministic template registry and slot filler.
- Storage boundary: typed local-store interface with in-memory and SQLite implementation boundaries.
- Adapters: model inference, mobile bindings, sync, and clinician tooling are explicit interfaces.

## Repository

    attune/
    ├── core/                 # Rust runtime
    ├── clinician-tools/      # Python template/evaluation tooling
    ├── mobile/               # framework-neutral TypeScript contract
    ├── templates/            # versioned clinician-authored template registry
    ├── demo/                 # zero-dependency browser MVP
    ├── docs/
    └── .github/workflows/

## Run the research stack

    cargo test --workspace
    cargo run -p attune-core --example demo
    python -m clinician_tools.validate_templates templates/templates.json

The default Rust build has no model downloads, network calls, or cloud dependencies.

## What comes after feedback

The expensive research phase is deliberately separated from the MVP:

1. Collect governed evaluation data.
2. Benchmark the current deterministic baseline.
3. Train and calibrate a crisis classifier.
4. Add real embedding retrieval and longitudinal modeling.
5. Run policy and response-model evaluations.
6. Spend GPU hours only on the components where the benchmark shows headroom.

## Safety implementation boundary

The bundled crisis detector is intentionally a conservative rule-based development implementation. The production INT8 classifier described by the product specification is represented by the CrisisModel trait and must be validated against a clinician-authored dataset before production use.

No claim about crisis recall, false-positive rate, clinical efficacy, or regulatory status is made by this repository.
