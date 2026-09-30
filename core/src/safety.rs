use crate::types::BeliefState;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use thiserror::Error;
#[derive(Debug,Clone,Copy,PartialEq,Eq)] pub enum SafetyDecision{Allow,Crisis,ScopeViolation}
#[derive(Debug,Clone)] pub struct SafetyResult{pub decision:SafetyDecision,pub input_hash:String,pub resource_injected:bool}
#[derive(Debug,Error)] pub enum SafetyError{#[error("safety configuration is invalid")]InvalidConfiguration}
pub trait CrisisModel:Send+Sync{fn crisis_probability(&self,input:&str)->f32;}
pub struct NoOpCrisisModel; impl CrisisModel for NoOpCrisisModel{fn crisis_probability(&self,_:&str)->f32{0.0}}
pub struct RuleCrisisDetector{crisis_terms:HashSet<&'static str>,scope_terms:HashSet<&'static str>}
impl Default for RuleCrisisDetector{fn default()->Self{Self{crisis_terms:["kill myself","suicide","end my life","self harm","hurt myself","overdose","want to die","abuse me"].into_iter().collect(),scope_terms:["diagnose me","what medication should i take","medication dosage","how much medication","prescribe"].into_iter().collect()}}}
impl RuleCrisisDetector{pub fn detect(&self,input:&str)->SafetyDecision{let n=input.trim().to_lowercase();if self.crisis_terms.iter().any(|x|n.contains(x)){SafetyDecision::Crisis}else if self.scope_terms.iter().any(|x|n.contains(x)){SafetyDecision::ScopeViolation}else{SafetyDecision::Allow}}}
/// Lightweight contextual detector used beside deterministic rules.
pub struct LearnedCrisisDetector{weights:std::collections::HashMap<String,f32>,bias:f32}
impl Default for LearnedCrisisDetector{fn default()->Self{let mut weights=std::collections::HashMap::new();for(t,w) in [("suicide",0.98),("kill",0.72),("myself",0.42),("die",0.62),("overdose",0.91),("selfharm",0.95),("hurt",0.32),("tonight",0.16),("plan",0.28),("means",0.22)]{weights.insert(t.to_string(),w);}Self{weights,bias:-1.35}}}
impl CrisisModel for LearnedCrisisDetector{fn crisis_probability(&self,input:&str)->f32{let n=input.to_lowercase().replace(['-','/','_']," ");let compact=n.replace(' ',"");let mut z=self.bias;for(t,w)in&self.weights{if n.split_whitespace().any(|x|x==t)||compact.contains(t){z+=*w;}}(1.0/(1.0+(-z).exp())).clamp(0.0,1.0)}}
pub struct SafetyShield<M:CrisisModel=NoOpCrisisModel>{rules:RuleCrisisDetector,model:M,crisis_threshold:f32}
impl<M:CrisisModel> SafetyShield<M>{pub fn new(model:M,threshold:f32)->Result<Self,SafetyError>{if !(0.0..=1.0).contains(&threshold){return Err(SafetyError::InvalidConfiguration)}Ok(Self{rules:RuleCrisisDetector::default(),model,crisis_threshold:threshold})}pub fn pre_filter(&self,input:&str)->SafetyResult{let h=hash_input(input);let rule=self.rules.detect(input);if rule==SafetyDecision::Crisis{return SafetyResult{decision:rule,input_hash:h,resource_injected:true}}if self.model.crisis_probability(input)>=self.crisis_threshold{return SafetyResult{decision:SafetyDecision::Crisis,input_hash:h,resource_injected:true}}SafetyResult{decision:rule,input_hash:h,resource_injected:false}}pub fn post_filter(&self,id:&str,registered:bool)->bool{registered&&!id.is_empty()}pub fn apply_risk(&self,state:&mut BeliefState,result:&SafetyResult){state.risk_flag=matches!(result.decision,SafetyDecision::Crisis);state.clamp()}}
pub fn hash_input(input:&str)->String{let mut h=Sha256::new();h.update(input.as_bytes());let d=h.finalize();format!("{:02x}{:02x}{:02x}{:02x}",d[0],d[1],d[2],d[3])}
pub fn crisis_text()->&'static str{"I’m not a crisis service. If you may be in immediate danger or might hurt yourself, contact local emergency services or a local crisis service now. If you can, move toward another person you trust. Try the 5-4-3-2-1 grounding exercise while you get support."}
pub fn scope_refusal_text()->&'static str{"I can help you reflect on what you’re experiencing, but I can’t diagnose conditions or advise on medication or dosage. A qualified clinician or pharmacist can help with those questions."}
