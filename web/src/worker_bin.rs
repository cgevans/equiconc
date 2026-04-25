//! Worker-side entry point. Trunk builds this as a separate wasm
//! module and emits a glue script that the worker URL points at; the
//! main thread spawns it via `WorkerBridgeBuilder`.
//!
//! All of the actual solver work — including `solve_with_progress`
//! and the throttled progress reporter — lives in
//! [`equiconc_web::worker`] so the binary stays at "register and run".

use equiconc_web::worker::SolverWorker;
use gloo_worker::Registrable;

fn main() {
    console_error_panic_hook::set_once();
    SolverWorker::registrar().register();
}
