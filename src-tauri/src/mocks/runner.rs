//! Async runner that drives mock progress events.

use std::collections::HashMap;
use std::sync::Mutex;
use tokio_util::sync::CancellationToken;

#[derive(Default)]
pub struct Runner {
    tokens: Mutex<HashMap<String, CancellationToken>>,
}

impl Runner {
    pub fn register(&self, id: String) -> CancellationToken {
        let t = CancellationToken::new();
        self.tokens.lock().unwrap().insert(id, t.clone());
        t
    }
    pub fn cancel(&self, id: &str) -> bool {
        if let Some(t) = self.tokens.lock().unwrap().remove(id) {
            t.cancel();
            true
        } else {
            false
        }
    }
    #[allow(dead_code)]
    pub fn forget(&self, id: &str) {
        self.tokens.lock().unwrap().remove(id);
    }
}
