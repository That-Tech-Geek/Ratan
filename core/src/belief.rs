use crate::types::{BeliefState, CheckInType, Readiness};

#[derive(Debug, Clone)]
pub struct Kalman2 {
    x: [f64; 2],
    p: [[f64; 2]; 2],
    q: [[f64; 2]; 2],
    r: f64,
}

impl Default for Kalman2 {
    fn default() -> Self {
        Self {
            x: [0.0, 0.5],
            p: [[0.25, 0.0], [0.0, 0.25]],
            q: [[0.01, 0.0], [0.0, 0.01]],
            r: 0.15,
        }
    }
}

impl Kalman2 {
    pub fn update(&mut self, observation: [f64; 2]) -> [f64; 2] {
        self.p[0][0] += self.q[0][0];
        self.p[1][1] += self.q[1][1];
        let k0 = self.p[0][0] / (self.p[0][0] + self.r);
        let k1 = self.p[1][1] / (self.p[1][1] + self.r);
        self.x[0] += k0 * (observation[0] - self.x[0]);
        self.x[1] += k1 * (observation[1] - self.x[1]);
        self.p[0][0] *= 1.0 - k0;
        self.p[1][1] *= 1.0 - k1;
        self.x
    }
}

#[derive(Debug, Clone)]
pub struct BeliefEngine {
    filter: Kalman2,
    alliance_samples: u32,
    alliance: f64,
    readiness: Readiness,
    recent_engagement: Vec<f64>,
}

impl Default for BeliefEngine {
    fn default() -> Self {
        Self {
            filter: Kalman2::default(),
            alliance_samples: 0,
            alliance: 0.5,
            readiness: Readiness::Contemplation,
            recent_engagement: Vec::new(),
        }
    }
}

impl BeliefEngine {
    pub fn state(&self, risk_flag: bool) -> BeliefState {
        let x = self.filter.x;
        let momentum = if self.recent_engagement.is_empty() {
            0.0
        } else {
            self.recent_engagement.iter().sum::<f64>() / self.recent_engagement.len() as f64
        };
        BeliefState {
            valence: x[0].clamp(-1.0, 1.0),
            arousal: x[1].clamp(0.0, 1.0),
            readiness: self.readiness,
            alliance: self.alliance.clamp(0.0, 1.0),
            risk_flag,
            session_momentum: momentum.clamp(-1.0, 1.0),
        }
    }

    pub fn submit_checkin(&mut self, kind: CheckInType, value: u8) {
        let v = (value.clamp(1, 5) as f64 - 3.0) / 2.0;
        match kind {
            CheckInType::Mood => {
                self.filter.update([v, (value as f64 - 1.0) / 4.0]);
            }
            CheckInType::DidThatLand => self.recent_engagement.push(v),
            CheckInType::AllianceHeard | CheckInType::AllianceDirection => {
                self.alliance_samples += 1;
                let observed = (value as f64 - 1.0) / 4.0;
                self.alliance = 0.7 * self.alliance + 0.3 * observed;
            }
        }
        if self.recent_engagement.len() > 3 {
            let drop = self.recent_engagement.len() - 3;
            self.recent_engagement.drain(0..drop);
        }
    }

    pub fn set_readiness(&mut self, readiness: Readiness) {
        self.readiness = readiness;
    }
}
