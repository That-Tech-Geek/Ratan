use attune_core::{AttuneRuntime, CheckInType};

fn main() {
    let mut runtime = AttuneRuntime::new(attune_core::safety::NoOpCrisisModel);

    runtime.submit_checkin(CheckInType::Mood, 2);
    let response = runtime.process_turn("I have had a difficult day.");
    println!("{} [{}] {}", response.template_id, response.move_id.unwrap_or_default(), response.rendered_text);

    let crisis = runtime.process_turn("I want to kill myself.");
    println!("{} crisis={} {}", crisis.template_id, crisis.crisis_triggered, crisis.rendered_text);

    let state = runtime.belief_state();
    println!("belief: valence={:.3} arousal={:.3} alliance={:.3}", state.valence, state.arousal, state.alliance);
}
