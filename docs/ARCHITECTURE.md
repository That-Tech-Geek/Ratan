# Attune implementation architecture

## Runtime invariant

The only legal execution path is:

1. input enters the pre-shield;
2. crisis/scope decisions can terminate the turn;
3. only clean input reaches belief/policy;
4. policy chooses a finite move under hard feasibility constraints;
5. expression renders only a registered template;
6. the post-shield validates the template ID before the response leaves core.

The policy engine has no API that accepts a crisis decision and no API that can emit arbitrary text.

## Current implementation

### Layer 0
- deterministic rule detector
- pluggable `CrisisModel` interface
- hashed input identifier for audit
- static crisis and scope responses
- registered-template post-filter
- dependence routing to M07 is enforced as a policy constraint

### Layer 1
- 2-state Kalman filter for valence/arousal
- explicit mood check-ins provide direct observations
- alliance and engagement updates are bounded
- readiness is an explicit state gate

### Layer 2
The current engine is a bounded contextual selector with seeded stochastic exploration. It uses move-specific expected reward, autonomy, exploration noise, feasibility masks, and a dependence penalty. The full Normal-Inverse-Gamma Bayesian posterior described by the product specification is deliberately not fabricated: its sufficient statistics and outcome-learning protocol need validated outcome data and persistence.

### Layer 3
- finite `MoveId` enum
- registry-backed templates
- closed slot set
- output length bound
- reviewer/version metadata
- no runtime generation

## Explicitly deferred

These require real external artifacts or clinical review and therefore remain interfaces rather than fake implementations:

- DistilBERT INT8 crisis classifier and measured recall
- MobileBERT sentiment model
- MiniLM embedding/HNSW semantic memory
- SQLCipher key management
- E2E sync
- clinician-authored 200+ crisis red-team corpus
- PHQ-9/GAD-7 trial protocol
- regulatory claims or certification

The repository should fail closed if those production integrations are missing rather than silently substituting an unvalidated model.
