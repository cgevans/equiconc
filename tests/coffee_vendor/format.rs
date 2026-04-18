#![allow(dead_code, clippy::all)]

// Vendored from COFFEE
// https://github.com/coffeesolverdev/coffee
// File: crates/coffee/src/format.rs
// See tests/coffee_vendor/LICENSE for the Apache-2.0 license text.
//
// Copyright 2025 coffeesolver.dev (UT Austin Capstone Team FH12 24-25)
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

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
