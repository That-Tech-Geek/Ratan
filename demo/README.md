# Attune MVP demo

This is a browser-only research preview for getting product and research feedback before spending GPU hours.

## Run

Open demo/index.html directly, or serve it locally:

    python -m http.server 8000 --directory demo

Then open http://localhost:8000.

## Real vs simulated

Real repository contracts: the Rust runtime contains the safety boundary, bounded state, memory, policy, personalization, deterministic generation and verification.

Demo shell: the browser uses a tiny dependency-free JavaScript implementation of the same interaction concepts so someone can try the product without Rust, Python packages, model weights, a GPU, a backend, or network access.

The demo is deliberately not presented as a trained model, clinical tool, or benchmark result. Its purpose is product discovery: interaction flow, response style, safety UX, useful signals, and feedback collection.

## Feedback questions

1. Does the interaction feel useful enough to use for 5 minutes?
2. Which runtime signal is useful or distracting?
3. Where does the safety boundary feel too strict or too loose?
4. What would make you trust the system with longitudinal context?
5. Which use case should be validated first?
