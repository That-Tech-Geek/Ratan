# Adaptive policy

`AdaptivePolicy` is an offline/contextual-bandit primitive. It learns rewards only for moves already admitted by the deterministic feasibility mask. It cannot override crisis handling, scope refusal, alliance constraints, or dependence controls.

The current reward model is deliberately simple. Production research should replace it with an offline-policy-evaluation pipeline before online adaptation is enabled.
