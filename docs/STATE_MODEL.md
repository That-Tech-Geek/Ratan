# Longitudinal state model

The latent state is an engineering representation of conversational signals, not a diagnosis or psychological truth. Each dimension is bounded and accompanied by uncertainty. Invalid observations are ignored and early observations receive a higher update weight.

The model is deliberately interpretable. A future learned temporal encoder can implement the same `observe`/`confidence` contract and must preserve uncertainty semantics.
