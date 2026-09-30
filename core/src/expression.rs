use crate::types::{BeliefState, MoveId, TemplateOutput};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct Template {
    pub template_id: String,
    pub move_id: MoveId,
    pub text: String,
    pub reviewer: String,
    pub review_date: String,
}

#[derive(Debug, Error)]
pub enum TemplateError {
    #[error("template not registered: {0}")]
    Unknown(String),
    #[error("invalid template output")]
    Invalid,
}

#[derive(Debug, Clone)]
pub struct TemplateRegistry {
    templates: HashMap<MoveId, Template>,
}

impl TemplateRegistry {
    pub fn from_templates(templates: Vec<Template>) -> Self {
        Self { templates: templates.into_iter().map(|t| (t.move_id, t)).collect() }
    }

    pub fn render(&self, move_id: MoveId, state: &BeliefState) -> Result<TemplateOutput, TemplateError> {
        let template = self.templates.get(&move_id).ok_or_else(|| TemplateError::Unknown(move_id.id().to_string()))?;
        let emotion = if state.valence < -0.5 { "overwhelmed" } else if state.valence < 0.0 { "frustrated" } else { "okay" };
        let rendered = template.text.replace("{emotion_word}", emotion);
        if rendered.len() > 2000 { return Err(TemplateError::Invalid); }
        Ok(TemplateOutput { template_id: template.template_id.clone(), rendered_text: rendered })
    }

    pub fn contains(&self, template_id: &str) -> bool {
        self.templates.values().any(|t| t.template_id == template_id)
    }
}

pub fn default_registry() -> TemplateRegistry {
    use MoveId::*;
    let make = |move_id: MoveId, text: &str| Template {
        template_id: format!("{}_V1", move_id.id()),
        move_id, text: text.to_string(), reviewer: "pending_clinician_review".into(), review_date: "unreviewed".into(),
    };
    TemplateRegistry::from_templates(vec![
        make(M01OpenReflection, "What feels most important to notice about this right now?"),
        make(M02Validation, "That sounds really {emotion_word}. It makes sense that this would feel heavy."),
        make(M03ExploratoryQuestion, "Would it help to look at what happened just before that feeling showed up?"),
        make(M04Grounding, "Let’s pause and orient to what is around you right now. Try naming five things you can see."),
        make(M05ThoughtRecord, "What thought showed up, and what evidence supports it or complicates it?"),
        make(M06BehavioralActivation, "What is one small, realistic action that would move you toward something you value?"),
        make(M07HumanConnection, "Who is one person you could connect with outside this app today?"),
        make(M08Psychoeducation, "I can share a short, general reflection on the pattern you described."),
        make(M09AgendaSetting, "What would make this conversation useful for you today?"),
        make(M10SummaryTask, "Here is a short summary of what you chose to focus on. What would you like to carry forward?"),
    ])
}
