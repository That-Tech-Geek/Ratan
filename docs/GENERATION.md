# Constrained generation

The generation layer is intentionally split into specification, realization and verification. A future local or remote LLM implements `Generator`; it receives only the bounded `ResponseSpec` and selected context.

Safety remains upstream and downstream. A generator cannot create a crisis response path, change policy constraints, or introduce forbidden scope content without the verifier rejecting it.
