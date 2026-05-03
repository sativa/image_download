//! Async runner that drives mock progress events. Filled in Task 7.2.

use std::collections::HashMap;
use std::sync::Mutex;
use tokio_util::sync::CancellationToken;

#[derive(Default)]
pub struct Runner {
    pub tokens: Mutex<HashMap<String, CancellationToken>>,
}
