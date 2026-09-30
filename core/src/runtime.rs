use crate::{
    belief::BeliefEngine,
    expression::{default_registry, TemplateRegistry},
    policy::{PolicyConfig, PolicyEngine},
    safety::{crisis_text, scope_refusal_text, CrisisModel, SafetyDecision, SafetyShield},
    types::{AuditEvent, BeliefState, CheckInType},
};
use serde::Serialize;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize)]
pub struct TurnResponse {
    pub template_id: String,
    pub rendered_text: String,
    pub move_id: Option<String>,
    pub crisis_triggered: bool,
    pub resource_injected: bool,
    pub input_hash: String,
}

pub struct AttuneRuntime<M: CrisisModel> {
    safety: SafetyShield<M>,
    belief: BeliefEngine,
    policy: PolicyEngine,
    templates: TemplateRegistry,
    turn_index: u32,
    sessions_last_14_days: u32,
    risk_latched: bool,
    audit: Vec<AuditEvent>,
}

impl<M: CrisisModel> AttuneRuntime<M> {
    pub fn new(model: M) -> Self {
        Self {
            safety: SafetyShield::new(model, 0.5).expect("static threshold is valid"),
            belief: BeliefEngine::default(),
            policy: PolicyEngine::new(PolicyConfig::default()),
            templates: default_registry(),
            turn_index: 0,
            sessions_last_14_days: 0,
            risk_latched: false,
            audit: Vec::new(),
        }
    }

    pub fn process_turn(&mut self, input: &str) -> TurnResponse {
        let safety = self.safety.pre_filter(input);
        match safety.decision {
            SafetyDecision::Crisis => {
                self.risk_latched = true;
                self.record("crisis", None, None, true, Some(safety.input_hash.clone()));
                return TurnResponse {
                    template_id: "SAFETY_CRISIS_V1".into(),
                    rendered_text: crisis_text().into(),
                    move_id: None,
                    crisis_triggered: true,
                    resource_injected: true,
                    input_hash: safety.input_hash,
                };
            }
            SafetyDecision::ScopeViolation => {
                self.record("scope_refusal", None, None, false, Some(safety.input_hash.clone()));
                return TurnResponse {
                    template_id: "SAFETY_SCOPE_V1".into(),
                    rendered_text: scope_refusal_text().into(),
                    move_id: None,
                    crisis_triggered: false,
                    resource_injected: false,
                    input_hash: safety.input_hash,
                };
            }
            SafetyDecision::Allow => {}
        }

        if self.risk_latched {
            self.record("risk_latched", None, None, true, Some(safety.input_hash.clone()));
            return TurnResponse {
                template_id: "SAFETY_CRISIS_V1".into(),
                rendered_text: crisis_text().into(),
                move_id: None,
                crisis_triggered: true,
                resource_injected: true,
                input_hash: safety.input_hash,
            };
        }

        let state = self.belief.state(false);
        let selected = self.policy.choose(&state, self.turn_index, self.sessions_last_14_days);
        let output = self.templates.render(selected, &state).expect("all default moves have templates");
        assert!(self.safety.post_filter(&output.template_id, self.templates.contains(&output.template_id)));
        self.record("move", Some(output.template_id.clone()), Some(selected.id().into()), false, Some(safety.input_hash.clone()));
        self.turn_index += 1;
        TurnResponse {
            template_id: output.template_id,
            rendered_text: output.rendered_text,
            move_id: Some(selected.id().into()),
            crisis_triggered: false,
            resource_injected: false,
            input_hash: safety.input_hash,
        }
    }

    pub fn submit_checkin(&mut self, kind: CheckInType, value: u8) {
        self.belief.submit_checkin(kind, value);
    }

    pub fn set_sessions_last_14_days(&mut self, count: u32) {
        self.sessions_last_14_days = count;
    }

    /// Start a new local session. A crisis latch is never cleared mid-session.
    /// A future supervised flow should gate session reset on its own explicit
    /// safety/review protocol rather than silently clearing the latch.
    pub fn start_new_session(&mut self) {
        self.turn_index = 0;
        self.risk_latched = false;
    }

    pub fn belief_state(&self) -> BeliefState {
        self.belief.state(false)
    }

    pub fn audit(&self) -> &[AuditEvent] {
        &self.audit
    }

    fn record(&mut self, event_type: &str, template_id: Option<String>, move_id: Option<String>, risk: bool, hash: Option<String>) {
        self.audit.push(AuditEvent {
            event_type: event_type.into(),
            timestamp_ms: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as i64,
            template_id,
            move_id,
            risk_triggered: risk,
            input_hash: hash,
        });
    }
}
