use attune_core::{AttuneRuntime, CheckInType};

#[test]
fn clean_turn_uses_registered_template() {
    let mut runtime = AttuneRuntime::new(attune_core::safety::NoOpCrisisModel);
    runtime.submit_checkin(CheckInType::Mood, 2);
    let response = runtime.process_turn("Today was hard.");
    assert!(!response.crisis_triggered);
    assert!(response.template_id.starts_with("M"));
    assert!(response.move_id.is_some());
}

#[test]
fn crisis_bypasses_policy() {
    let mut runtime = AttuneRuntime::new(attune_core::safety::NoOpCrisisModel);
    let response = runtime.process_turn("I want to end my life.");
    assert!(response.crisis_triggered);
    assert_eq!(response.template_id, "SAFETY_CRISIS_V1");
    assert!(response.move_id.is_none());
}

#[test]
fn scope_request_is_refused_without_policy_selection() {
    let mut runtime = AttuneRuntime::new(attune_core::safety::NoOpCrisisModel);
    let response = runtime.process_turn("what medication dosage should i take?");
    assert!(!response.crisis_triggered);
    assert_eq!(response.template_id, "SAFETY_SCOPE_V1");
    assert!(response.move_id.is_none());
}

#[test]
fn dependence_cap_routes_to_human_connection() {
    let mut runtime = AttuneRuntime::new(attune_core::safety::NoOpCrisisModel);
    runtime.set_sessions_last_14_days(11);
    let response = runtime.process_turn("I want to talk about something.");
    assert_eq!(response.move_id.as_deref(), Some("M07"));
}
