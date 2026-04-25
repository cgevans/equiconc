//! Main-thread handle around the gloo-worker bridge. Owns the worker
//! lifetime, exposes Leptos-friendly signals for solve state /
//! progress trace, and provides `start` and `cancel` callables.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use gloo_worker::{Spawnable, WorkerBridge};
use leptos::prelude::*;
use send_wrapper::SendWrapper;

use crate::state::{SolveError, SolveResult};
use crate::wire::{ProgressMsg, SolveRequest, SolveResponse};
use crate::worker::SolverWorker;

/// Path to the worker's wasm-bindgen JS glue emitted by Trunk. The
/// `data-bindgen-target="web"` attribute on the worker `<link>` makes
/// trunk emit the ES-module-style glue (`import init from ...`); gloo-
/// worker's default spawner (`as_module=true`, `with_loader=false`)
/// then wraps that glue in a Blob-URL `import init from PATH; await
/// init()` shim and spawns the Worker with `{ type: "module" }`.
///
/// Note: we deliberately do NOT use trunk's `data-loader-shim` here —
/// that emits a `importScripts(...) + wasm_bindgen(...)` shim against
/// the `--target no-modules` glue, which declares `wasm_bindgen` with
/// `let` (script-scoped) and breaks because the loader script can't
/// see the binding from the imported script. The module-worker path
/// has no such scoping issue.
const WORKER_PATH: &str = "./equiconc-web-worker.js";

/// Top-level solve state for the UI. The worker emits zero or more
/// `Progress` messages as it iterates, then transitions to `Done` or
/// `Failed`. `Done` boxes the `SolveResult` because it's much larger
/// than the other variants (carries the full `ndarray` matrices).
#[derive(Clone, Debug)]
pub enum SolveState {
    Idle,
    Running {
        iter: usize,
        gradient_norm: f64,
        elapsed_ms: f64,
    },
    Done(Box<SolveResult>),
    Failed(SolveError),
}

impl SolveState {
    pub fn is_running(&self) -> bool {
        matches!(self, SolveState::Running { .. })
    }
}

/// Type alias for the start callback's wire-Send shape (one `Arc` so
/// the handle is cheap to clone, one `SendWrapper` so it satisfies
/// Send+Sync without making the underlying `WorkerBridge` actually
/// thread-safe).
type StartCb = Arc<SendWrapper<Rc<dyn Fn(SolveRequest)>>>;
type CancelCb = Arc<SendWrapper<Rc<dyn Fn()>>>;

/// Handle exposed to the UI. `state` and `progress` are reactive
/// signals; `start` and `cancel` are callable from event handlers.
///
/// The two callables wrap an `Rc<dyn Fn>` (because the underlying
/// `gloo_worker::WorkerBridge` is `!Send`) inside `SendWrapper`, so the
/// whole struct can satisfy Leptos's `Send + Sync` bounds. Touching the
/// callables from any thread other than the one that constructed them
/// would panic, but wasm is single-threaded so this is safe.
#[derive(Clone)]
pub struct WorkerHandle {
    pub state: ReadSignal<SolveState>,
    pub progress_trace: ReadSignal<Vec<ProgressMsg>>,
    start_fn: StartCb,
    cancel_fn: CancelCb,
}

impl WorkerHandle {
    pub fn start(&self, req: SolveRequest) {
        (self.start_fn)(req);
    }
    pub fn cancel(&self) {
        (self.cancel_fn)();
    }
}

/// Construct a `WorkerHandle` and eagerly spawn the underlying worker
/// so the first Solve click doesn't pay the wasm-init cost. The bridge
/// is reused across solves (a stateless `SolverWorker::received` runs
/// each request from a fresh `equiconc::System`, so there's no state
/// to leak). Cancel terminates the worker and lazily re-spawns on the
/// next start.
pub fn install_worker() -> WorkerHandle {
    let (state, set_state) = signal(SolveState::Idle);
    let (progress, set_progress) = signal::<Vec<ProgressMsg>>(Vec::new());

    let bridge: Rc<RefCell<Option<WorkerBridge<SolverWorker>>>> = Rc::new(RefCell::new(None));

    // Eagerly spawn at install time so the wasm module starts loading
    // immediately on page load — by the time the user clicks Solve the
    // worker is ready and the UI doesn't hang on "starting…".
    *bridge.borrow_mut() = Some(spawn_bridge(set_state, set_progress));

    let bridge_for_start = Rc::clone(&bridge);
    let start_fn: Rc<dyn Fn(SolveRequest)> = Rc::new(move |req: SolveRequest| {
        // Reset trace + transition to a "starting" Running state.
        set_progress.set(Vec::new());
        set_state.set(SolveState::Running {
            iter: 0,
            gradient_norm: f64::NAN,
            elapsed_ms: 0.0,
        });

        // Lazily ensure the bridge exists, then drop the borrow before
        // sending so any synchronous re-entry from gloo-worker can take
        // its own borrow without panicking.
        if bridge_for_start.borrow().is_none() {
            let new_bridge = spawn_bridge(set_state, set_progress);
            *bridge_for_start.borrow_mut() = Some(new_bridge);
        }
        let slot = bridge_for_start.borrow();
        let bridge = slot.as_ref().expect("bridge present");
        bridge.send(req);
    });

    let bridge_for_cancel = Rc::clone(&bridge);
    let cancel_fn: Rc<dyn Fn()> = Rc::new(move || {
        // Dropping the bridge terminates the underlying Worker. The
        // next `start` call will eagerly re-spawn (paying the init
        // cost on the cancel-then-restart path, not the common one).
        *bridge_for_cancel.borrow_mut() = None;
        set_state.set(SolveState::Idle);
    });

    WorkerHandle {
        state,
        progress_trace: progress,
        start_fn: Arc::new(SendWrapper::new(start_fn)),
        cancel_fn: Arc::new(SendWrapper::new(cancel_fn)),
    }
}

/// Build a fresh `WorkerBridge` that routes worker responses back into
/// the supplied signals.
fn spawn_bridge(
    set_state: WriteSignal<SolveState>,
    set_progress: WriteSignal<Vec<ProgressMsg>>,
) -> WorkerBridge<SolverWorker> {
    SolverWorker::spawner()
        .callback(move |response: SolveResponse| match response {
            SolveResponse::Progress(p) => {
                let snapshot = p.clone();
                set_progress.update(|v| v.push(p));
                set_state.set(SolveState::Running {
                    iter: snapshot.iteration,
                    gradient_norm: snapshot.gradient_norm,
                    elapsed_ms: snapshot.elapsed_ms,
                });
            }
            SolveResponse::Done(wire) => {
                set_state.set(SolveState::Done(Box::new((*wire).into_result())));
            }
            SolveResponse::Error(wire) => {
                set_state.set(SolveState::Failed(wire.into_error()));
            }
        })
        .spawn(WORKER_PATH)
}
