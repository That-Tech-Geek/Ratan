//! Attune core runtime.
//!
//! The module boundaries mirror the product specification. Safety is deliberately
//! separated from policy: the policy engine receives only clean inputs.

pub mod belief;
pub mod expression;
pub mod policy;
pub mod runtime;
pub mod safety;
pub mod storage;
pub mod types;

pub use runtime::{AttuneRuntime, TurnResponse};
pub use types::{BeliefState, CheckInType, MoveId, Readiness};
