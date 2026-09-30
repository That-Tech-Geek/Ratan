//! Integrated research runtime facade. This wires all ten capability layers together
//! while retaining hard safety and scope arbitration.
use crate::{generation::{verify,Generator,ResponseSpec,Verification},learning::AdaptivePolicy,memory::SemanticMemory,multimodal::{fuse,CheckinSignals,VoiceSignals},personalization::{DependencySignals,PersonalizationEngine},safety::{CrisisModel,SafetyDecision,SafetyShield},state::LatentState,types::{BeliefState,MoveId}};
use crate::generation::TemplateGenerator;

pub struct ResearchRuntime<M:CrisisModel>{pub safety:SafetyShield<M>,pub memory:SemanticMemory,pub state:LatentState,pub policy:AdaptivePolicy,pub personalization:PersonalizationEngine,pub generator:Box<dyn Generator>}
impl<M:CrisisModel> ResearchRuntime<M>{
 pub fn new(model:M)->Self{Self{safety:SafetyShield::new(model,0.70).expect("valid threshold"),memory:SemanticMemory::default(),state:LatentState::default(),policy:AdaptivePolicy::default(),personalization:PersonalizationEngine::default(),generator:Box::new(TemplateGenerator)}}
 pub fn process(&mut self,input:&str,turn:u32)->String{let safety=self.safety.pre_filter(input);if matches!(safety.decision,SafetyDecision::Crisis){return crate::safety::crisis_text().into()}if matches!(safety.decision,SafetyDecision::ScopeViolation){return crate::safety::scope_refusal_text().into()}
  let mut belief=BeliefState::default();self.safety.apply_risk(&mut belief,&safety);if belief.risk_flag{return crate::safety::crisis_text().into()}
  self.memory.insert(format!("turn-{turn}"),input,turn as i64,0.4,0.0);let retrieved=self.memory.retrieve(input,turn as i64,3);let context=retrieved.first().map(|m|m.text.clone()).unwrap_or_default();
  let candidates:Vec<MoveId>=MoveId::ALL.iter().copied().filter(|m|self.personalization.allowed(m.id())).collect();let chosen=self.policy.select(&candidates,&belief);
  let spec=ResponseSpec{move_id:chosen,goal:"support and reflect".into(),tone:"warm and bounded".into(),max_chars:500,required:vec![],forbidden:vec!["diagnose".into(),"prescribe".into(),"I am your only".into()]};let out=self.generator.generate(&spec,&context);match verify(&spec,&out){Verification::Pass=>out,Verification::Reject(_)=>"I can stay with what you shared, one step at a time.".into()}
 }
 pub fn observe_modalities(&mut self,text:Option<CheckinSignals>,voice:Option<VoiceSignals>){let s=fuse(text,voice);self.state.observe(s.valence,s.arousal,0.5,0.5,s.confidence,0.5);}
 pub fn dependency_requires_connection(&self,s:DependencySignals)->bool{s.requires_connection()}
}
#[cfg(test)]mod tests{use super::*;use crate::safety::LearnedCrisisDetector;#[test]fn integrated_runtime_honors_crisis_boundary(){let mut r=ResearchRuntime::new(LearnedCrisisDetector::default());assert!(r.process("I want to kill myself",0).contains("crisis"));}#[test]fn integrated_runtime_returns_bounded_response(){let mut r=ResearchRuntime::new(LearnedCrisisDetector::default());let x=r.process("I feel overwhelmed by exams",1);assert!(!x.is_empty());assert!(x.len()<=500);}}
