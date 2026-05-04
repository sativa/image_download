//! CancellationToken registry shared between start_download and cancel_download.

use crate::commands::download::StartDownloadArgs;
use std::collections::HashMap;
use std::sync::Mutex;
use tokio_util::sync::CancellationToken;

#[derive(Default)]
pub struct Runner {
    tokens: Mutex<HashMap<String, CancellationToken>>,
    /// Stash of original args so retry_failed can re-run the same pipeline.
    args: Mutex<HashMap<String, StartDownloadArgs>>,
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
        self.args.lock().unwrap().remove(id);
    }
    pub fn stash_args(&self, id: String, args: StartDownloadArgs) {
        self.args.lock().unwrap().insert(id, args);
    }
    pub fn lookup_args(&self, id: &str) -> Option<StartDownloadArgs> {
        self.args.lock().unwrap().get(id).cloned()
    }
}
