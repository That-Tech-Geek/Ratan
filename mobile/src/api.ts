export type Readiness =
  | "precontemplation"
  | "contemplation"
  | "preparation"
  | "action"
  | "maintenance";

export interface BeliefState {
  valence: number;
  arousal: number;
  readiness: Readiness;
  alliance: number;
  risk_flag: boolean;
  session_momentum: number;
}

export interface TurnResponse {
  template_id: string;
  rendered_text: string;
  move_id: string | null;
  crisis_triggered: boolean;
  resource_injected: boolean;
  input_hash: string;
}

export interface AttuneCore {
  init(profileJson: string): Promise<void>;
  processTurn(input: string): Promise<TurnResponse>;
  getBeliefState(): Promise<BeliefState>;
  submitCheckin(type: string, value: number): Promise<void>;
  submitOutcome(instrument: string, scores: number[]): Promise<void>;
  exportAudit(): Promise<unknown>;
}
