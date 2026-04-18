#![allow(dead_code, clippy::all)]

// Vendored from COFFEE
// https://github.com/coffeesolverdev/coffee
// File: crates/coffee/src/extras.rs
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

use std::error::Error;
use std::fmt;

#[derive(Clone)]
pub struct OptimizerArgs {
    pub max_iterations: usize,
    pub max_delta: f64,
    pub eta: f64,
    pub norm_ratio_threshold: f64,
    pub rho_thresholds: [f64; 2],
    pub scale_factors: [f64; 2],
    pub use_terminal: bool,
    pub scalarity: bool,
    pub temp_celsius: f64,
    pub verbose: bool,
}

#[derive(Clone)]
pub struct OptimizerResults {
    pub optimal_x: Vec<f64>,
    pub optimal_lagrangian: f64,
    pub optimal_lambda: Vec<f64>,
    pub concentration_error: f64,
    pub log_messages: Vec<String>,
    pub elapsed_time: usize,
}

impl Default for OptimizerArgs {
    fn default() -> Self {
        OptimizerArgs {
            max_iterations: 250,
            max_delta: 1000.0,
            eta: 0.15,
            norm_ratio_threshold: 0.95,
            rho_thresholds: [0.25, 0.75],
            scale_factors: [0.25, 2.0],
            use_terminal: true,
            scalarity: true,
            temp_celsius: 37.0,
            verbose: false,
        }
    }
}

#[derive(Debug)]
pub struct OptimizerError(pub String);

impl fmt::Display for OptimizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for OptimizerError {}
