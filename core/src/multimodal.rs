//! Modality-neutral signal fusion. Signals are observations, never diagnoses.
#[derive(Debug,Clone,Copy,Default)]pub struct VoiceSignals{pub energy:f32,pub speech_rate:f32,pub pause_ratio:f32,pub confidence:f32}
#[derive(Debug,Clone,Copy,Default)]pub struct CheckinSignals{pub valence:f32,pub arousal:f32,pub confidence:f32}
#[derive(Debug,Clone,Copy,Default)]pub struct FusedSignal{pub valence:f32,pub arousal:f32,pub confidence:f32}
pub fn fuse(text:Option<CheckinSignals>,voice:Option<VoiceSignals>)->FusedSignal{let mut v=0.0;let mut a=0.0;let mut w=0.0;if let Some(t)=text{let c=t.confidence.clamp(0.0,1.0);v+=t.valence.clamp(-1.0,1.0)*c;a+=t.arousal.clamp(0.0,1.0)*c;w+=c;}if let Some(x)=voice{let c=x.confidence.clamp(0.0,1.0);let arousal=(0.5+0.35*(1.0-x.pause_ratio.clamp(0.0,1.0))+0.15*x.speech_rate.clamp(0.0,1.0)).clamp(0.0,1.0);a+=arousal*c;w+=c;}if w==0.0{return FusedSignal::default()}FusedSignal{valence:v/w,arousal:a/w,confidence:(w/2.0).min(1.0)}}
#[cfg(test)]mod tests{use super::*;#[test]fn fusion_preserves_uncertainty(){let s=fuse(Some(CheckinSignals{valence:-0.5,arousal:0.8,confidence:1.0}),None);assert!(s.confidence>0.49);assert!(s.valence<0.0);}}
