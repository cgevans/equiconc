#![allow(dead_code, clippy::all)]

// Vendored from COFFEE (Apache-2.0)
// https://github.com/coffeesolverdev/coffee
// File: crates/coffee/src/optimize.rs
//
// Modifications from upstream:
// - crate:: paths changed to super::
// - chrono::Utc replaced with std::time::Instant
// - Removed `use core::f64` (unnecessary in edition 2024)
// - Removed #[cfg(test)] module (we have our own tests)

use super::extras::{OptimizerArgs, OptimizerError, OptimizerResults};
use super::format::{conclude_message, process_message, start_message};
use super::steihaug::Steihaug;
use ndarray::{Array1, Array2, ArrayView1, Axis};
use std::error::Error;
use std::time::Instant;

const SMALLEST_EXP_VALUE: f64 = -230.0;

pub struct Optimizer {
    monomers: Array1<f64>,
    polymers: Array2<f64>,
    polymers_q: Array1<f64>,
    max_iterations: usize,
    curr_iteration: usize,
    time_us: usize,
    delta: f64,
    max_delta: f64,
    eta: f64,
    norm_ratio_threshold: f64,
    rho_thresholds: [f64; 2],
    scale_factors: [f64; 2],
    optimal_lambda: Array1<f64>,
    optimal_x: Array1<f64>,
    optimal_lagrangian: f64,
    steihaug_trust_region: Steihaug,
    use_terminal: bool,
    verbose: bool,
    log_msgs: Vec<String>,
    scalarity: bool,
    temp_celsius: f64,
}

fn density_water(t: f64) -> f64 {
    let a1 = -3.983035;
    let a2 = 301.797;
    let a3 = 522528.9;
    let a4 = 69.34881;
    let a5 = 999.974950;
    a5 * (1. - (t + a1) * (t + a1) * (t + a2) / a3 / (t + a4)) / 18.0152
}

impl Optimizer {
    pub fn new(
        monomers: &Array1<f64>,
        polymers: &Array2<f64>,
        polymers_q_nonexp: &Array1<f64>,
        optional_args: &OptimizerArgs,
    ) -> Result<Self, Box<dyn Error>> {
        let num_monomers = monomers.len();
        let num_polymers = polymers.len_of(Axis(0));

        if num_monomers == 0 {
            return Err(Box::new(OptimizerError(
                "Monomers array is empty.".to_string(),
            )));
        }
        if num_polymers == 0 {
            return Err(Box::new(OptimizerError(
                "Polymers array is empty.".to_string(),
            )));
        }
        if num_polymers < num_monomers {
            return Err(Box::new(OptimizerError(
                "Number of polymers is less than number of monomers.".to_string(),
            )));
        }

        if num_monomers != polymers.len_of(Axis(1)) {
            return Err(Box::new(OptimizerError(
                "Monomers and polymer compositions inconsistent.".to_string(),
            )));
        }
        if num_polymers != polymers_q_nonexp.len() {
            return Err(Box::new(OptimizerError(
                "Polymers and polymer quantities have different sizes.".to_string(),
            )));
        }

        let temp_celsius = optional_args.temp_celsius;
        let scalarity = optional_args.scalarity;
        let k_t = if scalarity {
            0.00198717 * (temp_celsius + 273.15)
        } else {
            1.0
        };
        let scaled_monomers = if scalarity {
            monomers / density_water(temp_celsius)
        } else {
            monomers.clone()
        };
        let polymers_q = polymers_q_nonexp.mapv(|x| (-x.max(SMALLEST_EXP_VALUE) / k_t).exp());

        let max_iterations = optional_args.max_iterations;
        Ok(Optimizer {
            monomers: scaled_monomers,
            polymers: polymers.clone(),
            polymers_q,
            max_iterations,
            curr_iteration: 0,
            time_us: 0,
            delta: 1.0,
            max_delta: optional_args.max_delta,
            eta: optional_args.eta,
            norm_ratio_threshold: optional_args.norm_ratio_threshold,
            rho_thresholds: optional_args.rho_thresholds,
            scale_factors: optional_args.scale_factors,
            optimal_lambda: Array1::zeros(num_monomers),
            optimal_x: Array1::zeros(num_polymers),
            optimal_lagrangian: 0.0,
            steihaug_trust_region: Steihaug::new(max_iterations, num_monomers),
            use_terminal: optional_args.use_terminal,
            verbose: optional_args.verbose,
            log_msgs: Vec::new(),
            scalarity,
            temp_celsius,
        })
    }

    fn norm(&self, v: ArrayView1<f64>) -> f64 {
        v.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    fn update_optimal_x(&mut self) {
        if self.scalarity {
            self.optimal_x =
                &self.polymers_q * &self.polymer_lambdas() * density_water(self.temp_celsius);
        } else {
            self.optimal_x = &self.polymers_q * &self.polymer_lambdas();
        }
    }

    fn polymer_lambdas(&self) -> Array1<f64> {
        (self.polymers.dot(&self.optimal_lambda)).exp()
    }

    fn lagrangian(&self, polymer_lambdas: &Array1<f64>) -> f64 {
        let after_energies = self.polymers_q.dot(polymer_lambdas);
        let after_initial = self.optimal_lambda.dot(&self.monomers);

        (after_energies - after_initial).ln()
    }

    fn jacobian(&self, polymer_lambdas: &Array1<f64>, lagrangian: f64) -> Array1<f64> {
        let after_energies = &self.polymers_q * polymer_lambdas;
        let jacobian = self.polymers.t().dot(&after_energies) - &self.monomers;
        jacobian / lagrangian.exp()
    }

    fn hessian(
        &self,
        polymer_lambdas: &Array1<f64>,
        lagrangian: f64,
        jacobian: &Array1<f64>,
    ) -> Array2<f64> {
        let first_part = 1. / lagrangian.exp();

        let after_energies = &self.polymers_q * polymer_lambdas;
        let polymerization = &self.polymers * &after_energies.insert_axis(Axis(1));

        let second_part = self.polymers.t().dot(&polymerization);

        let fourth_part = jacobian.view().insert_axis(Axis(1));

        let fifth_part = jacobian.view().insert_axis(Axis(0));

        first_part * second_part - fourth_part.dot(&fifth_part)
    }

    pub fn optimize(&mut self, initial_delta: f64) -> Result<bool, Box<dyn Error>> {
        if initial_delta <= 0.0 || !initial_delta.is_finite() {
            return Err(Box::new(OptimizerError(
                "Initial delta value is not valid.".to_string(),
            )));
        }

        self.print(&start_message());

        self.delta = initial_delta;
        let mut final_it = 0;
        self.reset();
        let start_time = Instant::now();

        for it in 0..self.max_iterations {
            let polymer_lambdas = self.polymer_lambdas();
            self.optimal_lagrangian = self.lagrangian(&polymer_lambdas);

            let function = self.optimal_lagrangian;
            let gradient = self.jacobian(&polymer_lambdas, self.optimal_lagrangian);
            let hessian = self.hessian(&polymer_lambdas, self.optimal_lagrangian, &gradient);

            let step = self.norm(gradient.view());
            let epsilon = step.sqrt().min(0.5f64) * step;

            let success = self
                .steihaug_trust_region
                .iterate(&gradient, &hessian, epsilon, self.delta);
            if !success {
                self.time_us = start_time.elapsed().as_micros() as usize;
                self.print(&conclude_message(
                    it,
                    success,
                    self.time_us,
                    self.verbose,
                    None,
                ));

                return Err(Box::new(OptimizerError(
                    "The Steihaug optimization did not succeed".to_string(),
                )));
            }
            let update_step = self.steihaug_trust_region.get_result();

            self.optimal_lambda = &self.optimal_lambda + &update_step;
            self.optimal_lagrangian = self.lagrangian(&self.polymer_lambdas());

            let pred_reduction =
                -(gradient.dot(&update_step) + 0.5 * update_step.dot(&hessian.dot(&update_step)));
            let actual_reduction = function - self.optimal_lagrangian;

            if actual_reduction == 0.0 {
                final_it = it;
                break;
            }

            let rho = if pred_reduction != 0.0 {
                actual_reduction / pred_reduction
            } else {
                0.0
            };

            if rho < self.rho_thresholds[0] {
                self.delta *= self.scale_factors[0];
            } else if rho > self.rho_thresholds[1]
                && self.norm(update_step.view()) >= self.norm_ratio_threshold * self.delta
            {
                self.delta = self.max_delta.min(self.scale_factors[1] * self.delta);
            }

            if rho <= self.eta {
                self.optimal_lambda = &self.optimal_lambda - &update_step;
                self.optimal_lagrangian = self.lagrangian(&self.polymer_lambdas());
            }

            self.update_optimal_x();
            self.print(&process_message(it, self.optimal_lagrangian, self.error()));

            final_it = it;
            self.curr_iteration += 1;
        }

        self.update_optimal_x();

        self.time_us = start_time.elapsed().as_micros() as usize;

        self.print(&conclude_message(
            final_it,
            true,
            self.time_us,
            self.verbose,
            Some(&OptimizerResults {
                optimal_x: self.optimal_x.to_vec(),
                optimal_lagrangian: self.optimal_lagrangian,
                optimal_lambda: self.optimal_lambda.to_vec(),
                concentration_error: self.error(),
                log_messages: self.log_msgs.clone(),
                elapsed_time: self.time_us,
            }),
        ));

        Ok(true)
    }

    pub fn reset(&mut self) {
        self.curr_iteration = 0;
        self.time_us = 0;
        self.optimal_lambda.fill(0.);
        self.optimal_x.fill(0.);
        self.optimal_lagrangian = 0.0;
        self.log_msgs.clear();
    }

    pub fn get_results(&self) -> OptimizerResults {
        OptimizerResults {
            optimal_x: self.optimal_x.to_vec(),
            optimal_lagrangian: self.optimal_lagrangian,
            optimal_lambda: self.optimal_lambda.to_vec(),
            concentration_error: self.error(),
            log_messages: self.log_msgs.clone(),
            elapsed_time: self.time_us,
        }
    }

    fn print(&mut self, msg: &str) {
        if self.use_terminal {
            print!("{}", msg);
        } else {
            self.log_msgs.push(msg.to_string());
        }
    }

    pub fn benchmark(&self) -> usize {
        self.time_us
    }

    pub fn iterations(&self) -> usize {
        self.curr_iteration
    }

    fn error(&self) -> f64 {
        let concs = self
            .polymers
            .t()
            .dot(&self.optimal_x.view().insert_axis(Axis(1)));
        let scaling = if self.scalarity {
            density_water(self.temp_celsius)
        } else {
            1.0
        };
        let backtrack = (&self.monomers * scaling).insert_axis(Axis(1)) - concs;
        backtrack
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| f64::max(a, b.abs()))
    }
}
