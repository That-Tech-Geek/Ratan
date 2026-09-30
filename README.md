# Attune

Privacy-first, on-device mental wellness companion built around a deterministic safety boundary, deterministic belief state, bounded contextual bandit policy, and clinician-authored template expression layer.

> Engineering preview. This repository is an implementation scaffold and research harness, not a clinical device and not a substitute for professional care.

## Architecture

`mobile -> runtime -> safety -> belief -> policy -> expression`

- **Layer 0:** hard pre/post safety shield. Vetoed inputs/actions never reach the policy selector.
- **Layer 1:** deterministic belief state with bounded valence/arousal Kalman updates, alliance, readiness, risk, and session momentum.
- **Layer 2:** bounded contextual Thompson sampling with explicit utility penalties and feasibility masks.
- **Layer 3:** deterministic template registry and slot filler. No runtime text generation.
- **Storage boundary:** typed local-store interface with an in-memory implementation for tests and a SQLite implementation boundary.
- **Adapters:** model inference, mobile bindings, sync, and clinician tooling are explicit interfaces rather than hidden dependencies.

## Repository

```text
attune/
├── core/                 # Rust runtime
│   ├── src/
│   └── Cargo.toml
├── clinician-tools/      # Python template/evaluation tooling
├── mobile/               # framework-neutral TypeScript contract
├── templates/            # versioned clinician-authored template registry
├── docs/
├── tests/
├── Cargo.toml
└── .github/workflows/ci.yml
```

## Run

```bash
cargo test --workspace
cargo run -p attune-core --example demo
python -m clinician_tools.validate_templates templates/templates.json
```

The default Rust build has no model downloads, network calls, or cloud dependencies.

## Safety implementation boundary

The bundled crisis detector is intentionally a conservative rule-based development implementation. The production INT8 classifier described by the product specification is represented by the `CrisisModel` trait and must be validated against a clinician-authored dataset before production use.

No claim about crisis recall, false-positive rate, clinical efficacy, or regulatory status is made by this repository.
