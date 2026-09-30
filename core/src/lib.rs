//! Attune core runtime.
//!
//! The module boundaries mirror the product specification. Safety is deliberately
//! separated from policy: the policy engine receives only clean inputs.

pub mod api;
pub mod belief;
pub mod expression;
pub mod policy;
pub mod runtime;
pub mod safety;\npub mod evaluation;\npub mod memory;\npub mod state;\npub mod learning;\npub mod generation;\npub mod personalization;\npub mod privacy;
pub mod storage;
pub mod types;

pub use runtime::{AttuneRuntime, TurnResponse};
pub use types::{BeliefState, CheckInType, MoveId, Readiness};
