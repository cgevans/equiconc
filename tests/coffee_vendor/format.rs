#![allow(dead_code, clippy::all)]

// Vendored from COFFEE (Apache-2.0)
// https://github.com/coffeesolverdev/coffee
// File: crates/coffee/src/format.rs

use super::extras::OptimizerResults;

pub fn start_message() -> String {
    "Starting COFFEE optimization...\r\n".to_string()
}

pub fn process_message(it: usize, lag: f64, error: f64) -> String {
    format!(
        "Iteration {}: f = {:.12}, error = {:.6e}\r\n",
        it, lag, error
    )
}

pub fn conclude_message(
    it: usize,
    success: bool,
    time_us: usize,
    display_time: bool,
    results: Option<&OptimizerResults>,
) -> String {
    let mut msg1 = format!(
        "Optimization {} after {} iterations.\r\n\r\n",
        if success { "complete" } else { "failed" },
        it
    );

    if let Some(results) = results {
        msg1.push_str(&format!(
            "Number of monomers: {}\nNumber of polymers: {}\r\n\r\n",
            results.optimal_lambda.len(),
            results.optimal_x.len()
        ));

        msg1.push_str(&format!(
            "Optimal Lagrangian: {:.6e}\r\n\r\n",
            results.optimal_lagrangian
        ));

        msg1.push_str("Optimal Lambdas:\r\n");
        for l_val in results.optimal_lambda.iter() {
            msg1.push_str(&format!("{:.6e} ", l_val));
        }
        msg1.push_str("\r\n\r\n");

        msg1.push_str(&format!(
            "Concentration Constraint Error: {:.6e}\r\n",
            results.concentration_error
        ));
    }
    if display_time {
        let et = time_us as f64 / 1000.0;
        if et < 1000.0 {
            return format!("{}\r\nElapsed time: {:.2} ms\r\n", msg1, et);
        } else {
            return format!("{}\r\nElapsed time: {:.2} s\r\n", msg1, et / 1000.0);
        }
    }
    msg1
}

pub fn results_message(results: &OptimizerResults) -> String {
    let mut msg = String::new();

    for x_val in results.optimal_x.iter() {
        msg.push_str(&format!("{:.2e} ", x_val));
    }
    msg
}
