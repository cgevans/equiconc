//! Output panels: diagnostics row, sortable concentrations table,
//! charts, and copy/download exports.

use leptos::prelude::*;
use leptos::wasm_bindgen::JsCast;
use leptos::web_sys;

use crate::charts::{ConcentrationsPie, MassShareBars};
use crate::exports::{report_json, results_csv, results_tsv};
use crate::state::{SolveError, SolveResult};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SortBy {
    Concentration,
    Index,
}

#[component]
pub fn OutputPanels(
    result: Signal<Option<SolveResult>>,
    error: Signal<Option<SolveError>>,
    cfe_text: RwSignal<String>,
    con_text: RwSignal<String>,
) -> impl IntoView {
    let (sort_by, set_sort_by) = signal(SortBy::Concentration);
    let (descending, set_descending) = signal(true);

    let toggle_sort = move |key: SortBy| {
        if sort_by.get() == key {
            set_descending.update(|d| *d = !*d);
        } else {
            set_sort_by.set(key);
            set_descending.set(matches!(key, SortBy::Concentration));
        }
    };

    view! {
        {move || error.get().map(|err| view! {
            <div class="error-box">
                <strong>{format_err_label(&err)}</strong> ": " {err.message.clone()}
            </div>
        })}

        {move || result.get().map(|res| {
            let order = sorted_order(&res, sort_by.get(), descending.get());
            let monomer_totals = res.mass_per_monomer();
            let shares: Vec<_> = (0..res.n_mon).map(|i| res.share_of_monomer(i)).collect();
            view! {
                <Diagnostics result=res.clone() />
                <ExportRow result=res.clone() cfe_text=cfe_text con_text=con_text />
                <table class="results">
                    <thead>
                        <tr>
                            <th on:click=move |_| toggle_sort(SortBy::Index)>"#"</th>
                            <th>"Name"</th>
                            <th class="num">"ΔG (kcal/mol)"</th>
                            <th class="num" on:click=move |_| toggle_sort(SortBy::Concentration)>
                                "[c] (M)"
                            </th>
                            {(0..res.n_mon).map(|i| view! {
                                <th class="num">{format!("share M{i}")}</th>
                            }).collect::<Vec<_>>()}
                        </tr>
                    </thead>
                    <tbody>
                        {order.iter().map(|&j| {
                            let is_mon = j < res.n_mon;
                            let name = if is_mon { format!("M{j}") } else { format!("S{j}") };
                            let row_shares = (0..res.n_mon).map(|i| {
                                let s = shares[i][j];
                                view! {
                                    <td class="num">{format!("{:.3}", s)}</td>
                                }
                            }).collect::<Vec<_>>();
                            view! {
                                <tr class:monomer=is_mon>
                                    <td class="num">{j.to_string()}</td>
                                    <td class="name">{name}</td>
                                    <td class="num">{format!("{:.3}", res.dg_kcal_used[j])}</td>
                                    <td class="num">{format!("{:.3e}", res.concentrations[j])}</td>
                                    {row_shares}
                                </tr>
                            }
                        }).collect::<Vec<_>>()}
                    </tbody>
                    <tfoot>
                        <tr>
                            <td colspan="3" style="text-align:right; color:var(--fg-muted);">
                                "Σ A·c (mass per monomer):"
                            </td>
                            <td></td>
                            {(0..res.n_mon).map(|i| view! {
                                <td class="num" style="color:var(--fg-muted);">
                                    {format!("{:.3e}", monomer_totals[i])}
                                </td>
                            }).collect::<Vec<_>>()}
                        </tr>
                    </tfoot>
                </table>

                <div class="chart-row">
                    <ConcentrationsPie result=result />
                    <MassShareBars result=result />
                </div>
            }
        })}
    }
}

fn format_err_label(err: &SolveError) -> String {
    use crate::state::ErrSource::*;
    match err.source {
        Composition => "Composition input".into(),
        Concentrations => "Concentrations input".into(),
        Solver => "Solver error".into(),
        Options => "Options".into(),
    }
}

#[component]
fn Diagnostics(result: SolveResult) -> impl IntoView {
    view! {
        <div class="diagnostics">
            <span><span class="label">"monomers: "</span><span class="value">{result.n_mon}</span></span>
            <span><span class="label">"species: "</span><span class="value">{result.concentrations.len()}</span></span>
            <span><span class="label">"iterations: "</span><span class="value">{result.iterations}</span></span>
            <span><span class="label">"converged: "</span><span class="value">{if result.converged_fully {"full"} else {"relaxed"}}</span></span>
            <span><span class="label">"max |c0 - Aᵀc|: "</span><span class="value">{format!("{:.2e}", result.residual)}</span></span>
            <span><span class="label">"elapsed: "</span><span class="value">{format!("{:.2} ms", result.elapsed_ms)}</span></span>
        </div>
    }
}

#[component]
fn ExportRow(
    result: SolveResult,
    cfe_text: RwSignal<String>,
    con_text: RwSignal<String>,
) -> impl IntoView {
    let result_for_tsv = result.clone();
    let result_for_csv = result.clone();
    let result_for_json = result.clone();
    view! {
        <div style="display:flex; gap:0.5rem; flex-wrap:wrap; margin: 0.5rem 0;">
            <button on:click=move |_| {
                copy_to_clipboard(&results_tsv(&result_for_tsv));
            }>"Copy TSV"</button>
            <button on:click=move |_| {
                trigger_download("equiconc-results.csv", "text/csv", &results_csv(&result_for_csv));
            }>"Download CSV"</button>
            <button on:click=move |_| {
                let cfe = cfe_text.get_untracked();
                let con = con_text.get_untracked();
                let payload = report_json(&cfe, &con, &result_for_json);
                trigger_download("equiconc-report.json", "application/json", &payload);
            }>"Download JSON report"</button>
        </div>
    }
}

fn sorted_order(result: &SolveResult, by: SortBy, desc: bool) -> Vec<usize> {
    let n = result.concentrations.len();
    let mut idx: Vec<usize> = (0..n).collect();
    match by {
        SortBy::Concentration => {
            idx.sort_by(|&a, &b| {
                let ca = result.concentrations[a];
                let cb = result.concentrations[b];
                if desc {
                    cb.partial_cmp(&ca).unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
                }
            });
        }
        SortBy::Index => {
            if desc {
                idx.reverse();
            }
        }
    }
    idx
}

fn copy_to_clipboard(s: &str) {
    let Some(window) = web_sys::window() else {
        return;
    };
    let clipboard = window.navigator().clipboard();
    let _ = clipboard.write_text(s);
}

fn trigger_download(filename: &str, mime: &str, content: &str) {
    let Some(window) = web_sys::window() else {
        return;
    };
    let Some(document) = window.document() else {
        return;
    };

    let array = js_sys::Array::new();
    array.push(&leptos::wasm_bindgen::JsValue::from_str(content));

    let opts = web_sys::BlobPropertyBag::new();
    opts.set_type(mime);
    let blob = match web_sys::Blob::new_with_str_sequence_and_options(&array, &opts) {
        Ok(b) => b,
        Err(_) => return,
    };

    let url = match web_sys::Url::create_object_url_with_blob(&blob) {
        Ok(u) => u,
        Err(_) => return,
    };

    let anchor = match document.create_element("a") {
        Ok(a) => a,
        Err(_) => return,
    };
    let anchor: web_sys::HtmlAnchorElement = match anchor.dyn_into() {
        Ok(a) => a,
        Err(_) => return,
    };
    anchor.set_href(&url);
    anchor.set_download(filename);
    let _ = anchor.set_attribute("style", "display:none");
    if let Some(body) = document.body() {
        let _ = body.append_child(&anchor);
        anchor.click();
        let _ = body.remove_child(&anchor);
    }
    let _ = web_sys::Url::revoke_object_url(&url);
}
