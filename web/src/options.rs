//! Tiered solver-options panel: Basic / Advanced / Expert collapsibles
//! plus two preset buttons.

use equiconc::SolverObjective;
use leptos::prelude::*;
use leptos::wasm_bindgen::JsCast;
use leptos::web_sys;

use crate::state::{EnergyUnit, TempUnit, UiOptions};

#[component]
pub fn OptionsPanel(options: RwSignal<UiOptions>) -> impl IntoView {
    view! {
        <div class="options">
            <div class="preset-row">
                <strong>"Presets:"</strong>
                <button on:click=move |_| options.set(UiOptions::equiconc_default())>
                    "equiconc defaults"
                </button>
                <button on:click=move |_| options.set(UiOptions::coffee_compatible())>
                    "COFFEE-compatible"
                </button>
            </div>
            <BasicOptions options=options />
            <details>
                <summary>"Advanced"</summary>
                <AdvancedOptions options=options />
            </details>
            <details>
                <summary>"Expert"</summary>
                <ExpertOptions options=options />
            </details>
        </div>
    }
}

// ---------------------------------------------------------------------------
// Basic
// ---------------------------------------------------------------------------

#[component]
fn BasicOptions(options: RwSignal<UiOptions>) -> impl IntoView {
    // Temperature is only needed when the energy unit is kcal/mol
    // (drives the RT divide) or when scalarity is on (water density
    // for the c₀ rescale). With RT-units inputs and scalarity off the
    // value isn't read by the solver, so we hide the row entirely.
    let show_temperature = move || {
        let o = options.get();
        o.energy_unit == EnergyUnit::KcalPerMol || o.scalarity
    };
    let clamp_units_label = move || match options.get().energy_unit {
        EnergyUnit::KcalPerMol => "ΔG clamp (kcal/mol)",
        EnergyUnit::RT => "ΔG clamp (RT units)",
    };
    view! {
        <div class="options-group" style="margin-top:0.4rem;">
            <label>
                <span>"Energy units"</span>
                <select
                    on:change=move |ev| {
                        let v = ev_string(&ev);
                        options.update(|o| {
                            o.energy_unit = match v.as_str() {
                                "RT" => EnergyUnit::RT,
                                _ => EnergyUnit::KcalPerMol,
                            };
                        });
                    }
                    prop:value=move || match options.get().energy_unit {
                        EnergyUnit::KcalPerMol => "kcal/mol",
                        EnergyUnit::RT => "RT",
                    }.to_string()
                >
                    <option value="kcal/mol">"kcal/mol"</option>
                    <option value="RT">"RT units (dimensionless ΔG/RT)"</option>
                </select>
            </label>
            {move || show_temperature().then(|| view! {
                <label>
                    <span>"Temperature"</span>
                    <span style="display:flex; gap:0.25rem;">
                        {sci_input_inline(
                            options,
                            |o| o.temperature_value,
                            |o, v| o.temperature_value = v,
                            "flex:1",
                        )}
                        <select
                            on:change=move |ev| {
                                let unit = ev_string(&ev);
                                options.update(|o| {
                                    o.temperature_unit = match unit.as_str() {
                                        "K" => TempUnit::Kelvin,
                                        _ => TempUnit::Celsius,
                                    };
                                });
                            }
                            prop:value=move || options.get().temperature_unit.label().to_string()
                        >
                            <option value="°C">"°C"</option>
                            <option value="K">"K"</option>
                        </select>
                    </span>
                </label>
            })}
            <label>
                <span>"Standard energy convention"</span>
                <select
                    on:change=move |ev| {
                        let v = ev_string(&ev);
                        options.update(|o| {
                            o.scalarity = matches!(v.as_str(), "fraction");
                        });
                    }
                    prop:value=move || if options.get().scalarity { "fraction" } else { "molar" }.to_string()
                >
                    <option value="molar">"1 M reference (ΔG° relative to 1 mol/L)"</option>
                    <option value="fraction">"Amount fraction (NUPACK / COFFEE convention)"</option>
                </select>
            </label>
            <label class="check">
                <input
                    type="checkbox"
                    prop:checked=move || options.get().dg_clamp_on
                    on:change=move |ev| {
                        let v = ev_checked(&ev);
                        options.update(|o| o.dg_clamp_on = v);
                    }
                />
                <span>{move || clamp_units_label().to_string()}</span>
            </label>
            {sci_field(options, "ΔG clamp value",
                |o| o.dg_clamp_kcal,
                |o, v| o.dg_clamp_kcal = v)}
            {int_field(options, "Max iterations",
                |o| o.solver.max_iterations as f64,
                |o, v| o.solver.max_iterations = v.max(1.0).round() as usize)}
            {sci_field_with_setter(options, "Convergence tolerance (relative)",
                |o| o.solver.gradient_rel_tol,
                |o, v| {
                    if !v.is_finite() || v <= 0.0 { return; }
                    o.solver.gradient_rel_tol = v;
                    // Scale absolute tolerance with the relative knob.
                    o.solver.gradient_abs_tol = v * 1e-15;
                    // Keep the relaxed-fallback floors at least as loose
                    // as the equiconc defaults: when the user tightens
                    // the strict tolerance below what's reachable in
                    // f64, the solver should still return at the
                    // relaxed tolerance with `converged_fully = false`
                    // rather than erroring out. (Validation also
                    // requires relaxed >= strict.)
                    o.solver.relaxed_gradient_rel_tol = (v * 1e3).max(1e-4);
                    o.solver.relaxed_gradient_abs_tol = (v * 1e-7).max(1e-14);
                })}
        </div>
    }
}

// ---------------------------------------------------------------------------
// Advanced
// ---------------------------------------------------------------------------

#[component]
fn AdvancedOptions(options: RwSignal<UiOptions>) -> impl IntoView {
    view! {
        <div class="options-group">
            <label>
                <span>"Objective"</span>
                <select
                    on:change=move |ev| {
                        let v = ev_string(&ev);
                        options.update(|o| {
                            o.solver.objective = match v.as_str() {
                                "Log" => SolverObjective::Log,
                                _ => SolverObjective::Linear,
                            };
                        });
                    }
                    prop:value=move || match options.get().solver.objective {
                        SolverObjective::Linear => "Linear",
                        SolverObjective::Log => "Log",
                    }.to_string()
                >
                    <option value="Linear">"Linear (default; convex)"</option>
                    <option value="Log">"Log (faster on stiff systems)"</option>
                </select>
            </label>
            {sci_field(options, "gradient_abs_tol",
                |o| o.solver.gradient_abs_tol,
                |o, v| o.solver.gradient_abs_tol = v)}
            {sci_field(options, "gradient_rel_tol",
                |o| o.solver.gradient_rel_tol,
                |o, v| o.solver.gradient_rel_tol = v)}
            {sci_field(options, "relaxed_gradient_abs_tol",
                |o| o.solver.relaxed_gradient_abs_tol,
                |o, v| o.solver.relaxed_gradient_abs_tol = v)}
            {sci_field(options, "relaxed_gradient_rel_tol",
                |o| o.solver.relaxed_gradient_rel_tol,
                |o, v| o.solver.relaxed_gradient_rel_tol = v)}
            {sci_field(options, "log_q_clamp (0 = off)",
                |o| o.solver.log_q_clamp.unwrap_or(0.0),
                |o, v| o.solver.log_q_clamp = if v > 0.0 { Some(v) } else { None })}
            {sci_field(options, "initial_trust_region_radius",
                |o| o.solver.initial_trust_region_radius,
                |o, v| o.solver.initial_trust_region_radius = v)}
            {sci_field(options, "max_trust_region_radius",
                |o| o.solver.max_trust_region_radius,
                |o, v| o.solver.max_trust_region_radius = v)}
        </div>
    }
}

// ---------------------------------------------------------------------------
// Expert
// ---------------------------------------------------------------------------

#[component]
fn ExpertOptions(options: RwSignal<UiOptions>) -> impl IntoView {
    view! {
        <div class="options-group">
            {sci_field(options, "step_accept_threshold",
                |o| o.solver.step_accept_threshold,
                |o, v| o.solver.step_accept_threshold = v)}
            {sci_field(options, "trust_region_shrink_rho",
                |o| o.solver.trust_region_shrink_rho,
                |o, v| o.solver.trust_region_shrink_rho = v)}
            {sci_field(options, "trust_region_grow_rho",
                |o| o.solver.trust_region_grow_rho,
                |o, v| o.solver.trust_region_grow_rho = v)}
            {sci_field(options, "trust_region_shrink_scale",
                |o| o.solver.trust_region_shrink_scale,
                |o, v| o.solver.trust_region_shrink_scale = v)}
            {sci_field(options, "trust_region_grow_scale",
                |o| o.solver.trust_region_grow_scale,
                |o, v| o.solver.trust_region_grow_scale = v)}
            {int_field(options, "stagnation_threshold",
                |o| o.solver.stagnation_threshold as f64,
                |o, v| o.solver.stagnation_threshold = v.max(1.0).round() as u32)}
            {sci_field(options, "log_c_clamp",
                |o| o.solver.log_c_clamp,
                |o, v| o.solver.log_c_clamp = v)}
        </div>
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Render the value `v` for use as input text, preferring the more
/// readable form. Pattern: scientific notation for very small / very
/// large magnitudes (< 1e-3 or ≥ 1e7), plain decimal otherwise.
fn format_value(v: f64) -> String {
    if v == 0.0 {
        return "0".into();
    }
    if !v.is_finite() {
        return format!("{v}");
    }
    let abs = v.abs();
    if !(1e-3..1e7).contains(&abs) {
        format!("{v:e}")
    } else {
        format!("{v}")
    }
}

/// A scientific-notation-friendly numeric input bound to one field of
/// `UiOptions`. Uses `type="text"` (not `type="number"`) so the browser
/// doesn't reformat partial entries — typing `1e-1` then `1` to mean
/// `1e-11` works as expected. We maintain a per-field text signal that
/// holds whatever the user is currently typing; it commits to the
/// option signal as soon as the text parses to a finite f64.
///
/// External changes to the option (e.g. the preset buttons) overwrite
/// the displayed text. The check `parsed != Some(v)` prevents the
/// effect from clobbering the user's mid-edit text on the very common
/// case where their last keystroke produced a valid number that
/// already matches the option.
fn sci_field<G, S>(
    options: RwSignal<UiOptions>,
    label: &'static str,
    get: G,
    set: S,
) -> impl IntoView
where
    G: Fn(&UiOptions) -> f64 + Copy + 'static,
    S: Fn(&mut UiOptions, f64) + Copy + 'static,
{
    sci_field_with_setter(options, label, get, move |o, v| {
        if v.is_finite() {
            set(o, v);
        }
    })
}

/// Like [`sci_field`], but the setter receives the parsed `v` and may
/// perform additional reactive bookkeeping (e.g. coupling secondary
/// tolerance fields to a primary one).
fn sci_field_with_setter<G, S>(
    options: RwSignal<UiOptions>,
    label: &'static str,
    get: G,
    set: S,
) -> impl IntoView
where
    G: Fn(&UiOptions) -> f64 + Copy + 'static,
    S: Fn(&mut UiOptions, f64) + Copy + 'static,
{
    view! {
        <label>
            <span>{label.to_string()}</span>
            {sci_input(options, get, set, "")}
        </label>
    }
}

/// Inline (no surrounding `<label>`) variant for use when the input
/// shares a row with another control (e.g. the temperature unit
/// dropdown). Returns just the `<input>`.
fn sci_input_inline<G, S>(
    options: RwSignal<UiOptions>,
    get: G,
    set: S,
    style: &'static str,
) -> impl IntoView
where
    G: Fn(&UiOptions) -> f64 + Copy + 'static,
    S: Fn(&mut UiOptions, f64) + Copy + 'static,
{
    sci_input(
        options,
        get,
        move |o, v| {
            if v.is_finite() {
                set(o, v);
            }
        },
        style,
    )
}

fn sci_input<G, S>(
    options: RwSignal<UiOptions>,
    get: G,
    set: S,
    style: &'static str,
) -> impl IntoView
where
    G: Fn(&UiOptions) -> f64 + Copy + 'static,
    S: Fn(&mut UiOptions, f64) + Copy + 'static,
{
    let initial = format_value(get(&options.get_untracked()));
    let text = RwSignal::new(initial);

    // External change → overwrite text. We compare via re-parsing
    // because the user's text and the formatted value may differ
    // representationally even when numerically equal (e.g. "1e-11"
    // vs "0.00000000001"); we don't want to clobber the user's
    // representation while they're still editing.
    Effect::new(move |_| {
        let v = get(&options.get());
        let parsed = text.get_untracked().trim().parse::<f64>().ok();
        if parsed != Some(v) {
            text.set(format_value(v));
        }
    });

    view! {
        <input
            type="text"
            inputmode="decimal"
            autocomplete="off"
            spellcheck="false"
            style=style
            prop:value=move || text.get()
            on:input=move |ev| {
                let raw = ev_string(&ev);
                text.set(raw.clone());
                if let Ok(v) = raw.trim().parse::<f64>() {
                    options.update(|o| set(o, v));
                }
            }
        />
    }
}

/// Integer field — same text-buffered behaviour, but rounds to integer.
fn int_field<G, S>(
    options: RwSignal<UiOptions>,
    label: &'static str,
    get: G,
    set: S,
) -> impl IntoView
where
    G: Fn(&UiOptions) -> f64 + Copy + 'static,
    S: Fn(&mut UiOptions, f64) + Copy + 'static,
{
    let initial = format!("{}", get(&options.get_untracked()).round() as i64);
    let text = RwSignal::new(initial);

    Effect::new(move |_| {
        let v = get(&options.get()).round() as i64;
        let parsed = text.get_untracked().trim().parse::<i64>().ok();
        if parsed != Some(v) {
            text.set(format!("{v}"));
        }
    });

    view! {
        <label>
            <span>{label.to_string()}</span>
            <input
                type="text"
                inputmode="numeric"
                autocomplete="off"
                spellcheck="false"
                prop:value=move || text.get()
                on:input=move |ev| {
                    let raw = ev_string(&ev);
                    text.set(raw.clone());
                    if let Ok(n) = raw.trim().parse::<i64>() {
                        options.update(|o| set(o, n as f64));
                    }
                }
            />
        </label>
    }
}

fn ev_string(ev: &web_sys::Event) -> String {
    let Some(target) = ev.target() else {
        return String::new();
    };
    if let Ok(input) = target.clone().dyn_into::<web_sys::HtmlInputElement>() {
        return input.value();
    }
    if let Ok(sel) = target.dyn_into::<web_sys::HtmlElement>()
        && let Ok(input) = sel.dyn_into::<web_sys::HtmlInputElement>()
    {
        return input.value();
    }
    String::new()
}

fn ev_checked(ev: &web_sys::Event) -> bool {
    let Some(target) = ev.target() else {
        return false;
    };
    target
        .dyn_into::<web_sys::HtmlInputElement>()
        .map(|i| i.checked())
        .unwrap_or(false)
}
