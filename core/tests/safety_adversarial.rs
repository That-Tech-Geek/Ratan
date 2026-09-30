use attune_core::safety::{LearnedCrisisDetector,SafetyDecision,SafetyShield};
#[test]
fn ensemble_detects_obfuscated_crisis(){let s=SafetyShield::new(LearnedCrisisDetector::default(),0.70).unwrap();assert_eq!(s.pre_filter("I want to kill-myself tonight").decision,SafetyDecision::Crisis);}
#[test]
fn ordinary_distress_is_not_forced_into_crisis(){let s=SafetyShield::new(LearnedCrisisDetector::default(),0.70).unwrap();assert_eq!(s.pre_filter("I had a stressful day").decision,SafetyDecision::Allow);}
