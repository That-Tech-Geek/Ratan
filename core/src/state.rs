//! Uncertainty-aware longitudinal user state.
#[derive(Debug,Clone,Copy,PartialEq)]
pub struct LatentState { pub valence:f32,pub arousal:f32,pub stress:f32,pub readiness:f32,pub engagement:f32,pub agency:f32,pub uncertainty:f32,pub observations:u32 }
impl Default for LatentState{fn default()->Self{Self{valence:0.0,arousal:0.0,stress:0.0,readiness:0.5,engagement:0.5,agency:0.5,uncertainty:1.0,observations:0}}}
impl LatentState{
 pub fn observe(&mut self,valence:f32,arousal:f32,stress:f32,readiness:f32,engagement:f32,agency:f32){let values=[valence,arousal,stress,readiness,engagement,agency];if values.iter().any(|x|!x.is_finite()){return;}let a=if self.observations<3{0.45}else{0.20};self.valence=(1.0-a)*self.valence+a*valence.clamp(-1.0,1.0);self.arousal=(1.0-a)*self.arousal+a*arousal.clamp(0.0,1.0);self.stress=(1.0-a)*self.stress+a*stress.clamp(0.0,1.0);self.readiness=(1.0-a)*self.readiness+a*readiness.clamp(0.0,1.0);self.engagement=(1.0-a)*self.engagement+a*engagement.clamp(0.0,1.0);self.agency=(1.0-a)*self.agency+a*agency.clamp(0.0,1.0);self.observations+=1;self.uncertainty=(1.0/(1.0+self.observations as f32)).max(0.05);}
 pub fn confidence(&self)->f32{1.0-self.uncertainty}
}
#[cfg(test)]mod tests{use super::*;#[test]fn uncertainty_declines_with_observations(){let mut s=LatentState::default();let u=s.uncertainty;s.observe(0.2,0.4,0.5,0.5,0.5,0.5);assert!(s.uncertainty<u);}#[test]fn invalid_observation_is_ignored(){let mut s=LatentState::default();s.observe(f32::NAN,0.0,0.0,0.0,0.0,0.0);assert_eq!(s.observations,0);}}
