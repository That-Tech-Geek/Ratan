//! Attune core runtime.
//! Safety is separated from policy; adaptive and generative components never bypass hard gates.
pub mod api;
pub mod belief;
pub mod expression;
pub mod policy;
pub mod runtime;
pub mod safety;
pub mod evaluation;
pub mod memory;
pub mod state;
pub mod learning;
pub mod generation;
pub mod personalization;
pub mod privacy;
pub mod multimodal;
pub mod storage;
pub mod types;
pub mod research;
pub use runtime::{AttuneRuntime,TurnResponse};
pub use types::{BeliefState,CheckInType,MoveId,Readiness};
