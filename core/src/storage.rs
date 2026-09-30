use crate::types::{AuditEvent, BeliefState};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("storage operation failed: {0}")]
    Failed(String),
}

pub trait LocalStore {
    fn save_belief(&mut self, session_id: &str, state: &BeliefState) -> Result<(), StorageError>;
    fn append_audit(&mut self, event: AuditEvent) -> Result<(), StorageError>;
    fn audit(&self, session_id: &str) -> Result<Vec<AuditEvent>, StorageError>;
}

/// Test/development store. Production storage must implement this interface with
/// encrypted-at-rest storage and the schema in docs/DATA_MODEL.sql.
#[derive(Default)]
pub struct MemoryStore {
    beliefs: HashMap<String, BeliefState>,
    audits: Vec<AuditEvent>,
}

impl LocalStore for MemoryStore {
    fn save_belief(&mut self, session_id: &str, state: &BeliefState) -> Result<(), StorageError> {
        self.beliefs.insert(session_id.to_string(), state.clone());
        Ok(())
    }

    fn append_audit(&mut self, event: AuditEvent) -> Result<(), StorageError> {
        self.audits.push(event);
        Ok(())
    }

    fn audit(&self, _session_id: &str) -> Result<Vec<AuditEvent>, StorageError> {
        Ok(self.audits.clone())
    }
}
