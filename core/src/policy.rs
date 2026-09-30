use crate::types::{BeliefState, MoveId, Readiness};
use rand::{rngs::StdRng, Rng, SeedableRng};

#[derive(Debug, Clone, Copy)]
pub struct MoveSpec {
    pub id: MoveId,
    pub autonomy: f64,
}

pub const MOVE_SPECS: [MoveSpec; 10] = [
    MoveSpec { id: MoveId::M01OpenReflection, autonomy: 0.90 },
    MoveSpec { id: MoveId::M02Validation, autonomy: 0.85 },
    MoveSpec { id: MoveId::M03ExploratoryQuestion, autonomy: 0.80 },
    MoveSpec { id: MoveId::M04Grounding, autonomy: 0.75 },
    MoveSpec { id: MoveId::M05ThoughtRecord, autonomy: 0.60 },
    MoveSpec { id: MoveId::M06BehavioralActivation, autonomy: 0.50 },
    MoveSpec { id: MoveId::M07HumanConnection, autonomy: 0.95 },
    MoveSpec { id: MoveId::M08Psychoeducation, autonomy: 0.70 },
    MoveSpec { id: MoveId::M09AgendaSetting, autonomy: 0.80 },
    MoveSpec { id: MoveId::M10SummaryTask, autonomy: 0.70 },
];

#[derive(Debug, Clone)]
pub struct PolicyConfig {
    pub dependence_weight: f64,
    pub exploration: f64,
    pub seed: u64,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self { dependence_weight: 0.8, exploration: 0.15, seed: 42 }
    }
}

#[derive(Debug, Clone)]
pub struct PolicyEngine {
    config: PolicyConfig,
    rng: StdRng,
}

impl PolicyEngine {
    pub fn new(config: PolicyConfig) -> Self {
        Self { rng: StdRng::seed_from_u64(config.seed), config }
    }

    pub fn choose(&mut self, state: &BeliefState, turn_index: u32, sessions_last_14_days: u32) -> MoveId {
        assert!(!state.risk_flag, "policy must never receive a risk-flagged state");
        let candidates = MOVE_SPECS.iter().copied().filter(|m| feasible(m.id, state, turn_index));
        let mut best = MoveId::M02Validation;
        let mut best_score = f64::NEG_INFINITY;
        for spec in candidates {
            let expected = expected_reward(spec, state);
            let noise = self.rng.gen_range(-self.config.exploration..=self.config.exploration);
            let dependence = if sessions_last_14_days > 10 && spec.id != MoveId::M07HumanConnection {
                self.config.dependence_weight * ((sessions_last_14_days - 10) as f64 / 10.0).min(2.0)
            } else { 0.0 };
            let score = expected + noise + 0.25 * spec.autonomy - dependence;
            if score > best_score {
                best_score = score;
                best = spec.id;
            }
        }
        if sessions_last_14_days > 10 { MoveId::M07HumanConnection } else { best }
    }
}

fn expected_reward(spec: MoveSpec, state: &BeliefState) -> f64 {
    let state_term = match spec.id {
        MoveId::M04Grounding => state.arousal * 0.7,
        MoveId::M07HumanConnection => (1.0 - state.alliance) * 0.8,
        _ => (0.5 - state.valence.abs()) * 0.2,
    };
    state_term + spec.autonomy * 0.3
}

fn feasible(id: MoveId, state: &BeliefState, turn: u32) -> bool {
    if state.risk_flag { return false; }
    if state.alliance <= 0.5 && !matches!(id, MoveId::M01OpenReflection | MoveId::M02Validation | MoveId::M07HumanConnection | MoveId::M09AgendaSetting) {
        return false;
    }
    match id {
        MoveId::M03ExploratoryQuestion => state.readiness.at_least(Readiness::Contemplation),
        MoveId::M04Grounding => state.arousal > 0.6,
        MoveId::M05ThoughtRecord | MoveId::M06BehavioralActivation => state.readiness.at_least(Readiness::Preparation),
        MoveId::M08Psychoeducation => state.alliance > 0.5,
        MoveId::M09AgendaSetting => turn < 3,
        MoveId::M10SummaryTask => turn > 8,
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn early_turn_mask_is_hard() {
        let state = BeliefState::default();
        let mut p = PolicyEngine::new(PolicyConfig { seed: 1, ..Default::default() });
        for _ in 0..20 {
            let selected = p.choose(&state, 1, 0);
            assert!(matches!(selected, MoveId::M01OpenReflection | MoveId::M02Validation | MoveId::M07HumanConnection | MoveId::M09AgendaSetting));
        }
    }
}
