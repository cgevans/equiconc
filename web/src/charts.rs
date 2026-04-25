//! Hand-rolled SVG charts. We avoid pulling in a chart crate so the
//! wasm bundle stays small and we don't fight someone else's reactive
//! integration.

use leptos::prelude::*;
use std::f64::consts::TAU;

use crate::state::SolveResult;
use crate::wire::ProgressMsg;

/// Top-N concentrations rendered as a pie chart, with the remaining
/// species lumped into an "other" wedge. `n` includes the "other"
/// bucket — pass `n = 6` for top-5-plus-other.
#[component]
pub fn ConcentrationsPie(result: Signal<Option<SolveResult>>) -> impl IntoView {
    move || {
        let res = result.get()?;
        let total: f64 = res.concentrations.iter().sum();
        if total <= 0.0 || !total.is_finite() {
            return None;
        }

        let mut indexed: Vec<(usize, f64)> = res
            .concentrations
            .iter()
            .enumerate()
            .map(|(i, c)| (i, *c))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_n = 5;
        let mut wedges: Vec<(String, f64)> = Vec::new();
        let mut other = 0.0;
        for (rank, (idx, c)) in indexed.iter().enumerate() {
            if rank < top_n {
                let label = if *idx < res.n_mon {
                    format!("M{idx}")
                } else {
                    format!("S{idx}")
                };
                wedges.push((label, *c));
            } else {
                other += *c;
            }
        }
        if other > 0.0 {
            wedges.push(("other".into(), other));
        }

        let cx = 110.0_f64;
        let cy = 110.0_f64;
        let r = 100.0_f64;
        let mut path_views = Vec::new();
        let mut legend_views = Vec::new();
        let mut start = -TAU / 4.0; // 12 o'clock
        for (i, (label, c)) in wedges.iter().enumerate() {
            let frac = c / total;
            let end = start + frac * TAU;
            let large_arc = if frac > 0.5 { 1 } else { 0 };
            let x1 = cx + r * start.cos();
            let y1 = cy + r * start.sin();
            let x2 = cx + r * end.cos();
            let y2 = cy + r * end.sin();
            let d = format!(
                "M {cx},{cy} L {x1:.2},{y1:.2} A {r},{r} 0 {large_arc} 1 {x2:.2},{y2:.2} Z"
            );
            let color = palette(i);
            path_views
                .push(view! { <path d=d fill=color stroke="var(--bg-panel)" stroke-width="1" /> });

            let pct = frac * 100.0;
            legend_views.push(view! {
                <li>
                    <span style=format!("display:inline-block;width:0.7em;height:0.7em;background:{};margin-right:0.3em;border-radius:2px;", color)></span>
                    {label.clone()}
                    " — "
                    {format!("{pct:.1}%")}
                </li>
            });
            start = end;
        }

        Some(view! {
            <div class="chart">
                <h3 style="margin:0 0 0.5rem; font-size:0.95rem;">"Top species (by concentration)"</h3>
                <div style="display:flex; align-items:flex-start; gap:0.75rem;">
                    <svg viewBox="0 0 220 220" style="width:220px; height:220px; flex-shrink:0;">
                        {path_views}
                    </svg>
                    <ul style="list-style:none; padding:0; margin:0; font-size:0.85rem;">
                        {legend_views}
                    </ul>
                </div>
            </div>
        })
    }
}

/// Per-monomer share-of-mass bar chart. One stack per monomer; each
/// stack is split into the species containing that monomer.
#[component]
pub fn MassShareBars(result: Signal<Option<SolveResult>>) -> impl IntoView {
    move || {
        let res = result.get()?;
        let n_mon = res.n_mon;
        if n_mon == 0 {
            return None;
        }

        let bar_w = 60.0_f64;
        let gap = 24.0_f64;
        let plot_h = 200.0_f64;
        let label_h = 24.0_f64;
        let total_w = (bar_w + gap) * n_mon as f64 + gap;
        let total_h = plot_h + label_h + 12.0;

        let mut rects = Vec::new();
        let mut labels = Vec::new();
        for i in 0..n_mon {
            let share = res.share_of_monomer(i);
            let mut indexed: Vec<(usize, f64)> =
                share.iter().enumerate().map(|(j, s)| (j, *s)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let x = gap + i as f64 * (bar_w + gap);
            let mut y_cursor = label_h;
            for (rank, (j, s)) in indexed.iter().enumerate() {
                if *s <= 0.0 {
                    continue;
                }
                let h = s * plot_h;
                let color = palette(rank);
                let label = if *j < res.n_mon {
                    format!("M{j}")
                } else {
                    format!("S{j}")
                };
                let pct = s * 100.0;
                let title = format!("{label}: {pct:.1}%");
                rects.push(view! {
                    <rect
                        x={format!("{x:.2}")}
                        y={format!("{:.2}", y_cursor)}
                        width={format!("{bar_w:.2}")}
                        height={format!("{:.2}", h.max(0.0))}
                        fill=color
                        stroke="var(--bg-panel)"
                        stroke-width="0.5"
                    >
                        <title>{title}</title>
                    </rect>
                });
                y_cursor += h.max(0.0);
            }
            labels.push(view! {
                <text
                    x={format!("{:.2}", x + bar_w / 2.0)}
                    y={format!("{:.2}", label_h - 6.0)}
                    text-anchor="middle"
                    font-size="11"
                    fill="var(--fg)"
                >
                    {format!("M{i}")}
                </text>
            });
        }

        Some(view! {
            <div class="chart">
                <h3 style="margin:0 0 0.5rem; font-size:0.95rem;">"Share of monomer mass"</h3>
                <svg viewBox=format!("0 0 {total_w:.2} {total_h:.2}") style="width:100%; height:auto;">
                    {rects}
                    {labels}
                </svg>
            </div>
        })
    }
}

/// Color palette for chart slices. Stable across renders so the legend
/// and the wedges line up.
fn palette(i: usize) -> &'static str {
    const COLORS: &[&str] = &[
        "#0d6efd", "#dc3545", "#198754", "#ffc107", "#6f42c1", "#fd7e14", "#20c997", "#d63384",
        "#6610f2", "#0dcaf0",
    ];
    COLORS[i % COLORS.len()]
}

/// Convergence trace: log10 of the gradient norm vs. iteration number,
/// streamed live from the worker. Renders as an SVG polyline with
/// labelled axes; updates reactively as `progress_trace` grows.
#[component]
pub fn ConvergenceChart(
    progress_trace: ReadSignal<Vec<ProgressMsg>>,
    /// Optional final iteration to mark the converged endpoint when
    /// the worker has emitted a `Done`. Drawn as a small dot at the
    /// last point.
    final_iteration: Signal<Option<usize>>,
) -> impl IntoView {
    move || {
        let trace = progress_trace.get();
        if trace.len() < 2 {
            // Hide until we have at least two samples — a single-point
            // line is useless and the empty chart would shift layout.
            return None;
        }

        // Plot area in user-space units; outer SVG scales to fit.
        let pad_left = 44.0_f64;
        let pad_right = 12.0_f64;
        let pad_top = 18.0_f64;
        let pad_bottom = 28.0_f64;
        let plot_w = 360.0_f64;
        let plot_h = 160.0_f64;
        let total_w = pad_left + plot_w + pad_right;
        let total_h = pad_top + plot_h + pad_bottom;

        // y axis: log10(gradient_norm). Substitute a sentinel for
        // non-positive / non-finite values so the chart still renders.
        let log10s: Vec<f64> = trace
            .iter()
            .map(|p| {
                if p.gradient_norm > 0.0 && p.gradient_norm.is_finite() {
                    p.gradient_norm.log10()
                } else {
                    -300.0
                }
            })
            .collect();
        let y_min = log10s.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max = log10s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let y_span = (y_max - y_min).max(1.0);

        let x_max = (trace.len().max(2) - 1) as f64;

        let to_xy = |i: usize, log10y: f64| -> (f64, f64) {
            let x = pad_left + (i as f64 / x_max) * plot_w;
            // Invert y so larger gradient is at top.
            let y = pad_top + (1.0 - (log10y - y_min) / y_span) * plot_h;
            (x, y)
        };

        let mut points = String::new();
        for (i, &log10y) in log10s.iter().enumerate() {
            let (x, y) = to_xy(i, log10y);
            if i > 0 {
                points.push(' ');
            }
            points.push_str(&format!("{x:.2},{y:.2}"));
        }

        let last_dot = final_iteration.get().and_then(|_| {
            log10s
                .last()
                .map(|&y| to_xy(log10s.len() - 1, y))
                .map(|(x, y)| view! {
                    <circle cx=format!("{x:.2}") cy=format!("{y:.2}") r="3" fill="var(--success)" />
                })
        });

        let y_axis_label = format!("{y_max:.1}");
        let y_axis_label_min = format!("{y_min:.1}");
        let x_axis_label = format!("{}", trace.len());

        Some(view! {
            <div class="chart">
                <h3 style="margin:0 0 0.5rem; font-size:0.95rem;">"Convergence (log₁₀ ‖∇‖ vs. iteration)"</h3>
                <svg
                    viewBox=format!("0 0 {total_w:.2} {total_h:.2}")
                    style="width:100%; height:auto;"
                >
                    // Plot frame
                    <rect
                        x=format!("{pad_left:.2}")
                        y=format!("{pad_top:.2}")
                        width=format!("{plot_w:.2}")
                        height=format!("{plot_h:.2}")
                        fill="none"
                        stroke="var(--border)"
                        stroke-width="1"
                    />
                    // y-axis labels (top + bottom of plot)
                    <text
                        x=format!("{:.2}", pad_left - 6.0)
                        y=format!("{:.2}", pad_top + 4.0)
                        text-anchor="end"
                        font-size="11"
                        fill="var(--fg-muted)"
                    >
                        {y_axis_label}
                    </text>
                    <text
                        x=format!("{:.2}", pad_left - 6.0)
                        y=format!("{:.2}", pad_top + plot_h + 2.0)
                        text-anchor="end"
                        font-size="11"
                        fill="var(--fg-muted)"
                    >
                        {y_axis_label_min}
                    </text>
                    // x-axis label
                    <text
                        x=format!("{:.2}", pad_left + plot_w / 2.0)
                        y=format!("{:.2}", pad_top + plot_h + 18.0)
                        text-anchor="middle"
                        font-size="11"
                        fill="var(--fg-muted)"
                    >
                        {format!("iteration (0 → {x_axis_label})")}
                    </text>
                    // Trace
                    <polyline
                        fill="none"
                        stroke="var(--accent)"
                        stroke-width="1.5"
                        points=points
                    />
                    {last_dot}
                </svg>
            </div>
        })
    }
}
