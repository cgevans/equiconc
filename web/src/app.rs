//! Top-level layout. Owns the input/options signals and the
//! [`WorkerHandle`], and wires their state into the panels.

use leptos::prelude::*;

use crate::charts::ConvergenceChart;
use crate::inputs::InputPanels;
use crate::options::OptionsPanel;
use crate::outputs::OutputPanels;
use crate::state::UiOptions;
use crate::testcases::TESTCASES;
use crate::wire::{SolveRequest, WireUiOptions};
use crate::worker_handle::{SolveState, install_worker};

/// Throttle progress messages from the worker. 50 ms ≈ 20 fps —
/// fast enough to feel live, slow enough that a tight inner loop
/// can't drown the channel.
const PROGRESS_THROTTLE_MS: f64 = 50.0;

#[component]
pub fn App() -> impl IntoView {
    // Seed inputs with the first testcase so the first-time visitor
    // sees a working solve immediately on Solve.
    let initial = &TESTCASES[0];
    let cfe_text = RwSignal::new(initial.cfe.to_string());
    let con_text = RwSignal::new(initial.con.to_string());
    let options = RwSignal::new(UiOptions::default());

    // Worker handle is `!Send` (it owns an `Rc<RefCell<WorkerBridge>>`),
    // but everything in this CSR app runs on the main thread. We pass
    // it to closures by clone (each clone bumps the inner Rc).
    let worker = install_worker();

    let on_solve = {
        let worker = worker.clone();
        move |_| {
            if worker.state.get_untracked().is_running() {
                return;
            }
            let req = SolveRequest {
                cfe: cfe_text.get_untracked(),
                con: con_text.get_untracked(),
                options: WireUiOptions::from(&options.get_untracked()),
                progress_throttle_ms: PROGRESS_THROTTLE_MS,
            };
            worker.start(req);
        }
    };

    let on_cancel = {
        let worker = worker.clone();
        move |_| worker.cancel()
    };

    // Derived signals for the rest of the UI: project Done/Failed
    // arms out of SolveState into option-typed signals the panels
    // already understand. `Signal::derive` rather than `Memo::new`
    // because `SolveResult`/`SolveError` don't impl `PartialEq`.
    let result = {
        let worker = worker.clone();
        Signal::derive(move || match worker.state.get() {
            SolveState::Done(r) => Some(*r),
            _ => None,
        })
    };
    let error = {
        let worker = worker.clone();
        Signal::derive(move || match worker.state.get() {
            SolveState::Failed(e) => Some(e),
            _ => None,
        })
    };

    let state_for_button = worker.state;
    let state_for_cancel = worker.state;
    let state_for_status = worker.state;
    let progress_trace = worker.progress_trace;
    let final_iteration = {
        let worker = worker.clone();
        Signal::derive(move || match worker.state.get() {
            SolveState::Done(r) => Some(r.iterations),
            _ => None,
        })
    };

    view! {
        <main>
            <header>
                <h1>"equiconc"</h1>
            </header>

            <InputPanels cfe_text=cfe_text con_text=con_text />

            <OptionsPanel options=options />

            <div class="solve-bar">
                <button
                    class="primary"
                    on:click=on_solve
                    prop:disabled=move || state_for_button.get().is_running()
                >
                    {move || if state_for_button.get().is_running() { "Solving…" } else { "Solve" }}
                </button>

                {move || state_for_cancel.get().is_running().then(|| {
                    let on_cancel = on_cancel.clone();
                    view! { <button on:click=on_cancel>"Cancel"</button> }
                })}

                {move || match state_for_status.get() {
                    SolveState::Idle => view! { <span></span> }.into_any(),
                    SolveState::Running { iter, gradient_norm, elapsed_ms } => view! {
                        <span class="spinner" aria-label="solving"></span>
                        <span class="status">
                            {if iter == 0 {
                                // Worker hasn't sent the first iteration yet.
                                // Usually <1 frame with the persistent worker;
                                // the only time this lingers is the very first
                                // solve before the wasm module finishes loading.
                                "spawning worker…".to_string()
                            } else {
                                format!(
                                    "iter {iter}, ‖∇‖ = {gradient_norm:.2e}, {elapsed_ms:.0} ms",
                                )
                            }}
                        </span>
                    }.into_any(),
                    SolveState::Done(r) => view! {
                        <span class="status ok">
                            {format!("OK — {} species, {} iterations, {:.2} ms",
                                r.concentrations.len(), r.iterations, r.elapsed_ms)}
                        </span>
                    }.into_any(),
                    SolveState::Failed(e) => view! {
                        <span class="status error">{e.message}</span>
                    }.into_any(),
                }}
            </div>

            <OutputPanels
                result=result
                error=error
                cfe_text=cfe_text
                con_text=con_text
            />

            // Convergence chart sits last so it's the only post-solve-bar
            // element while the worker is still running (no result yet);
            // once `OutputPanels` materializes, this falls below the
            // results table where it serves as historical context rather
            // than the main visualization.
            <ConvergenceChart
                progress_trace=progress_trace
                final_iteration=final_iteration
            />

            <footer>
                "Solve runs entirely in this tab. No data is uploaded. "
                <a href="https://github.com/cgevans/equiconc" target="_blank" rel="noopener">
                    "Source on GitHub"
                </a>
                "."
            </footer>
        </main>
    }
}
