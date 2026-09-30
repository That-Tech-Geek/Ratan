//! Constrained contextual bandit for intervention selection.
//! Learning only ranks already-safe candidate moves; it cannot bypass safety masks.
use crate::types::{BeliefState,MoveId};
#[derive(Debug,Clone,Copy)]struct Arm{n:u32,mean:f64}
#[derive(Debug,Clone)]pub struct AdaptivePolicy{arms:[Arm;10],exploration:f64}
impl Default for AdaptivePolicy{fn default()->Self{Self{arms:[Arm{n:0,mean:0.0};10],exploration:0.35}}}
impl AdaptivePolicy{
 fn idx(id:MoveId)->usize{MoveId::ALL.iter().position(|x|*x==id).unwrap()}
 pub fn update(&mut self,id:MoveId,reward:f64){let a=&mut self.arms[Self::idx(id)];a.n+=1;a.mean+=(reward.clamp(-1.0,1.0)-a.mean)/a.n as f64;}
 pub fn select(&self,candidates:&[MoveId],state:&BeliefState)->MoveId{assert!(!state.risk_flag);let total=self.arms.iter().map(|a|a.n).sum::<u32>().max(1) as f64;candidates.iter().copied().max_by(|a,b|self.score(*a,total).total_cmp(&self.score(*b,total))).unwrap_or(MoveId::M02Validation)}
 fn score(&self,id:MoveId,total:f64)->f64{let a=self.arms[Self::idx(id)];let bonus=if a.n==0{self.exploration}else{self.exploration*(total.ln()/a.n as f64).sqrt()};a.mean+bonus}
 pub fn arm_stats(&self)->Vec<(MoveId,u32,f64)>{MoveId::ALL.iter().map(|id|{let a=self.arms[Self::idx(*id)];(*id,a.n,a.mean)}).collect()}
}
#[cfg(test)]mod tests{use super::*;#[test]fn learning_updates_only_selected_arm(){let mut p=AdaptivePolicy::default();p.update(MoveId::M03ExploratoryQuestion,0.8);let s=BeliefState::default();let chosen=p.select(&[MoveId::M03ExploratoryQuestion,MoveId::M02Validation],&s);assert!(matches!(chosen,MoveId::M03ExploratoryQuestion|MoveId::M02Validation));assert_eq!(p.arm_stats()[2].1,1);}}
