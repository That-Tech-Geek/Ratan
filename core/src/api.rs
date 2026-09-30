use crate::{safety::CrisisModel, types::{BeliefState, CheckInType}, AttuneRuntime, TurnResponse};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Profile {
    pub user_id: String,
    pub preferences: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditBundle {
    pub events: Vec<crate::types::AuditEvent>,
    pub belief_state: BeliefState,
}

/// Stable application-facing facade. Mobile bindings should wrap this surface
/// rather than reaching into policy or safety internals.
pub struct CoreApi<M: CrisisModel> {
    runtime: AttuneRuntime<M>,
    profile: Option<Profile>,
}

impl<M: CrisisModel> CoreApi<M> {
    pub fn new(model: M) -> Self {
        Self { runtime: AttuneRuntime::new(model), profile: None }
    }

    pub fn init(&mut self, profile_json: &str) -> Result<(), serde_json::Error> {
        self.profile = Some(serde_json::from_str(profile_json)?);
        Ok(())
    }

    pub fn process_turn(&mut self, input: &str) -> TurnResponse {
        self.runtime.process_turn(input)
    }

    pub fn get_belief_state(&self) -> BeliefState {
        self.runtime.belief_state()
    }

    pub fn submit_checkin(&mut self, kind: CheckInType, value: u8) {
        self.runtime.submit_checkin(kind, value);
    }

    pub fn export_audit(&self) -> AuditBundle {
        AuditBundle {
            events: self.runtime.audit().to_vec(),
            belief_state: self.runtime.belief_state(),
        }
    }

    pub fn start_new_session(&mut self) {
        self.runtime.start_new_session();
    }
}
