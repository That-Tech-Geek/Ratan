use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Readiness {
    Precontemplation,
    Contemplation,
    Preparation,
    Action,
    Maintenance,
}

impl Readiness {
    pub fn at_least(self, other: Self) -> bool {
        use Readiness::*;
        let rank = |x| match x {
            Precontemplation => 0,
            Contemplation => 1,
            Preparation => 2,
            Action => 3,
            Maintenance => 4,
        };
        rank(self) >= rank(other)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MoveId {
    M01OpenReflection,
    M02Validation,
    M03ExploratoryQuestion,
    M04Grounding,
    M05ThoughtRecord,
    M06BehavioralActivation,
    M07HumanConnection,
    M08Psychoeducation,
    M09AgendaSetting,
    M10SummaryTask,
}

impl MoveId {
    pub const ALL: [Self; 10] = [
        Self::M01OpenReflection, Self::M02Validation, Self::M03ExploratoryQuestion,
        Self::M04Grounding, Self::M05ThoughtRecord, Self::M06BehavioralActivation,
        Self::M07HumanConnection, Self::M08Psychoeducation, Self::M09AgendaSetting,
        Self::M10SummaryTask,
    ];

    pub fn id(self) -> &'static str {
        match self {
            Self::M01OpenReflection => "M01",
            Self::M02Validation => "M02",
            Self::M03ExploratoryQuestion => "M03",
            Self::M04Grounding => "M04",
            Self::M05ThoughtRecord => "M05",
            Self::M06BehavioralActivation => "M06",
            Self::M07HumanConnection => "M07",
            Self::M08Psychoeducation => "M08",
            Self::M09AgendaSetting => "M09",
            Self::M10SummaryTask => "M10",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CheckInType {
    Mood,
    DidThatLand,
    AllianceHeard,
    AllianceDirection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefState {
    pub valence: f64,
    pub arousal: f64,
    pub readiness: Readiness,
    pub alliance: f64,
    pub risk_flag: bool,
    pub session_momentum: f64,
}

impl Default for BeliefState {
    fn default() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.5,
            readiness: Readiness::Contemplation,
            alliance: 0.5,
            risk_flag: false,
            session_momentum: 0.0,
        }
    }
}

impl BeliefState {
    pub fn clamp(&mut self) {
        self.valence = self.valence.clamp(-1.0, 1.0);
        self.arousal = self.arousal.clamp(0.0, 1.0);
        self.alliance = self.alliance.clamp(0.0, 1.0);
        self.session_momentum = self.session_momentum.clamp(-1.0, 1.0);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateOutput {
    pub template_id: String,
    pub rendered_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub event_type: String,
    pub timestamp_ms: i64,
    pub template_id: Option<String>,
    pub move_id: Option<String>,
    pub risk_triggered: bool,
    pub input_hash: Option<String>,
}
