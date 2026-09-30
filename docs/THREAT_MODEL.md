# Threat model

## Assets
- conversation text and metadata
- embeddings and longitudinal profiles
- safety/audit events
- encryption keys handled by host platform

## Threats
1. local database theft
2. accidental sensitive-memory retrieval
3. prompt injection through stored memories
4. generator scope bypass
5. crisis detector evasion
6. dependency amplification
7. telemetry leakage

## Controls
- authenticated local encryption
- sensitivity-aware retrieval
- bounded response specifications
- post-generation verification
- deterministic safety rules plus contextual detector
- dependency guardrails
- no network telemetry in core

## Residual risk
The repository does not claim clinical validation, regulatory clearance, or production mobile keychain integration. Those require external review and platform-specific security testing.
