//! Input panels: composition (`.cfe` / `.ocx`) and concentrations
//! (`.con`). Each panel is a `<textarea>` with file-load and drag-drop
//! support.
//!
//! The textarea is the source of truth — we do not parse on every
//! keystroke. Validation happens at solve time. We never install a
//! `keydown` handler that captures `Tab`, so the browser-default focus
//! advance is preserved (pressing Tab moves focus out of the textarea,
//! which is the normal a11y behaviour for plain text inputs).

use leptos::prelude::*;
use leptos::wasm_bindgen::JsCast;
use leptos::web_sys;

use crate::testcases::{TESTCASES, Testcase};

#[component]
pub fn InputPanels(cfe_text: RwSignal<String>, con_text: RwSignal<String>) -> impl IntoView {
    view! {
        <div class="panels">
            <TextPanel
                title="Composition (.cfe / .ocx)"
                hint="One row per species: stoichiometry columns then ΔG (kcal/mol). NUPACK bookkeeping columns auto-detected."
                signal=cfe_text
                accept=".cfe,.ocx,.txt,.csv,.tsv"
                rows=14
            />
            <TextPanel
                title="Concentrations (.con)"
                hint="One total monomer concentration per line, in molar units (e.g. 1e-6)."
                signal=con_text
                accept=".con,.txt,.csv,.tsv"
                rows=14
            />
        </div>
        <TestcasePicker cfe_text=cfe_text con_text=con_text />
    }
}

#[component]
fn TextPanel(
    title: &'static str,
    hint: &'static str,
    signal: RwSignal<String>,
    accept: &'static str,
    rows: u32,
) -> impl IntoView {
    let (dragover, set_dragover) = signal_local(false);
    let textarea_ref: NodeRef<leptos::html::Textarea> = NodeRef::new();

    let on_input = move |ev: web_sys::Event| {
        if let Some(target) = ev.target()
            && let Ok(ta) = target.dyn_into::<web_sys::HtmlTextAreaElement>()
        {
            signal.set(ta.value());
        }
    };

    let on_drop = move |ev: web_sys::DragEvent| {
        ev.prevent_default();
        set_dragover.set(false);
        let dt = match ev.data_transfer() {
            Some(d) => d,
            None => return,
        };
        let files = match dt.files() {
            Some(f) => f,
            None => return,
        };
        if files.length() == 0 {
            return;
        }
        let Some(file) = files.get(0) else { return };
        load_file_into(signal, &file);
    };

    let on_dragover = move |ev: web_sys::DragEvent| {
        ev.prevent_default();
        set_dragover.set(true);
    };
    let on_dragleave = move |_| set_dragover.set(false);

    let on_pick = move |ev: web_sys::Event| {
        if let Some(target) = ev.target()
            && let Ok(input) = target.dyn_into::<web_sys::HtmlInputElement>()
        {
            if let Some(files) = input.files()
                && files.length() > 0
                && let Some(file) = files.get(0)
            {
                load_file_into(signal, &file);
            }
            input.set_value("");
        }
    };

    let on_clear = move |_| signal.set(String::new());

    view! {
        <section class="panel">
            <h2>
                {title.to_string()}
                <span class="panel-actions">
                    <label class="file-pick" title="Load file">
                        "Load…"
                        <input
                            type="file"
                            accept=accept
                            on:change=on_pick
                            style="display:none"
                        />
                    </label>
                    <button on:click=on_clear>"Clear"</button>
                </span>
            </h2>
            <p style="color:var(--fg-muted); font-size:0.8rem; margin:0 0 0.4rem;">
                {hint.to_string()}
            </p>
            <textarea
                node_ref=textarea_ref
                rows=rows
                spellcheck="false"
                prop:value=move || signal.get()
                on:input=on_input
                on:dragover=on_dragover
                on:dragleave=on_dragleave
                on:drop=on_drop
                class:dragover=move || dragover.get()
            />
        </section>
    }
}

fn load_file_into(signal: RwSignal<String>, file: &web_sys::File) {
    use leptos::wasm_bindgen::closure::Closure;

    let reader = match web_sys::FileReader::new() {
        Ok(r) => r,
        Err(_) => return,
    };
    let reader_clone = reader.clone();
    let onload = Closure::<dyn FnMut(web_sys::Event)>::new(move |_ev: web_sys::Event| {
        if let Ok(result) = reader_clone.result()
            && let Some(text) = result.as_string()
        {
            signal.set(text);
        }
    });
    reader.set_onload(Some(onload.as_ref().unchecked_ref()));
    onload.forget();
    let _ = reader.read_as_text(file);
}

#[component]
fn TestcasePicker(cfe_text: RwSignal<String>, con_text: RwSignal<String>) -> impl IntoView {
    let on_change = move |ev: web_sys::Event| {
        let Some(target) = ev.target() else { return };
        let Ok(sel) = target.dyn_into::<web_sys::HtmlInputElement>() else {
            return;
        };
        let value = sel.value();
        if value.is_empty() {
            return;
        }
        if let Ok(idx) = value.parse::<usize>()
            && let Some(tc) = TESTCASES.get(idx)
        {
            apply_testcase(tc, cfe_text, con_text);
        }
        sel.set_value("");
    };

    view! {
        <div style="margin-top:0.5rem; display:flex; align-items:center; gap:0.5rem;">
            <label style="color:var(--fg-muted); font-size:0.85rem;">
                "Load testcase: "
            </label>
            <select on:change=on_change>
                <option value="">"— pick one —"</option>
                {TESTCASES.iter().enumerate().map(|(i, tc)| view! {
                    <option value={i.to_string()} title={tc.note}>
                        {tc.name.to_string()}
                    </option>
                }).collect::<Vec<_>>()}
            </select>
        </div>
    }
}

fn apply_testcase(tc: &Testcase, cfe: RwSignal<String>, con: RwSignal<String>) {
    cfe.set(tc.cfe.to_string());
    con.set(tc.con.to_string());
}
