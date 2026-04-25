//! Solver worker. Receives a [`SolveRequest`] from the main thread,
//! drives `equiconc::System::solve_with_progress`, and emits a stream
//! of [`SolveResponse::Progress`] messages followed by exactly one
//! [`SolveResponse::Done`] or [`SolveResponse::Error`].
//!
//! Progress messages are throttled to at most one per
//! `progress_throttle_ms` milliseconds so a fast solve doesn't drown
//! the message channel. The first iteration is always emitted (so the
//! UI flips out of the "starting…" state immediately) and the final
//! iteration is implicit in the `Done` message that follows.

use equiconc::{IterationStatus, SolveControl};
use gloo_worker::{HandlerId, Worker, WorkerScope};
use leptos::wasm_bindgen::JsCast;

use crate::solver::solve_with_progress;
use crate::wire::{ProgressMsg, SolveErrorWire, SolveRequest, SolveResponse};

/// gloo-worker bridge for the solver. Stateless — every request runs
/// from a fresh `equiconc::System` so warm-start doesn't leak between
/// solves on the same worker. (We `terminate()` the worker on cancel
/// anyway, so this only matters across consecutive solves on a worker
/// the user lets run to completion.)
pub struct SolverWorker;

impl Worker for SolverWorker {
    type Message = ();
    type Input = SolveRequest;
    type Output = SolveResponse;

    fn create(_scope: &WorkerScope<Self>) -> Self {
        SolverWorker
    }

    fn update(&mut self, _scope: &WorkerScope<Self>, _msg: Self::Message) {}

    fn received(&mut self, scope: &WorkerScope<Self>, msg: Self::Input, who: HandlerId) {
        let started_at = perf_now();
        let throttle_ms = msg.progress_throttle_ms.max(0.0);
        let mut last_emit = f64::NEG_INFINITY;

        let scope_for_progress = scope.clone();
        let on_iter = |status: &IterationStatus| -> SolveControl {
            let now = perf_now();
            // Always emit the first iteration; afterwards, throttle.
            if status.iteration == 1 || now - last_emit >= throttle_ms {
                last_emit = now;
                scope_for_progress.respond(
                    who,
                    SolveResponse::Progress(ProgressMsg {
                        iteration: status.iteration,
                        gradient_norm: status.gradient_norm,
                        objective: status.objective,
                        trust_radius: status.trust_radius,
                        elapsed_ms: now - started_at,
                    }),
                );
            }
            SolveControl::Continue
        };

        let response = match solve_with_progress(&msg.cfe, &msg.con, &msg.options.into(), on_iter) {
            Ok(result) => SolveResponse::Done(Box::new(result.to_wire())),
            Err(err) => SolveResponse::Error(SolveErrorWire {
                message: err.message,
                source: err.source,
            }),
        };
        scope.respond(who, response);
    }
}

fn perf_now() -> f64 {
    // In a worker context `web_sys::window()` returns None; the
    // analogous global is `WorkerGlobalScope`.
    if let Some(window) = web_sys::window()
        && let Some(p) = window.performance()
    {
        return p.now();
    }
    if let Ok(scope) = js_sys::global().dyn_into::<web_sys::WorkerGlobalScope>()
        && let Some(p) = scope.performance()
    {
        return p.now();
    }
    0.0
}
