#![allow(dead_code, clippy::all, mismatched_lifetime_syntaxes)]

// Vendored from COFFEE (Apache-2.0)
// https://github.com/coffeesolverdev/coffee
// File: crates/coffee/src/steihaug.rs

use ndarray::{Array1, Array2, ArrayView1};

pub struct Steihaug {
    curr_iterations: usize,
    max_iterations: usize,
    vector_size: usize,
    curr_zstep: Array1<f64>,
    curr_rstep: Array1<f64>,
    curr_dstep: Array1<f64>,
}

impl Steihaug {
    pub fn new(max_iterations: usize, vector_size: usize) -> Self {
        Self {
            curr_iterations: 0,
            max_iterations,
            vector_size,
            curr_zstep: Array1::zeros(vector_size),
            curr_rstep: Array1::zeros(vector_size),
            curr_dstep: Array1::zeros(vector_size),
        }
    }

    fn norm(&self, v: ArrayView1<f64>) -> f64 {
        v.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    fn solve_curvature_quadratic(&self, delta: f64) -> Option<f64> {
        let a = self.curr_dstep.dot(&self.curr_dstep);
        let b = 2.0 * self.curr_zstep.dot(&self.curr_dstep);
        let c = self.curr_zstep.dot(&self.curr_zstep) - delta * delta;

        let t = (-b + (b * b - 4.0 * a * c).sqrt()) / (2.0 * a);

        if t.is_finite() {
            Some(t)
        } else {
            None
        }
    }

    fn early_update_zstep(&mut self, delta: f64) -> bool {
        let tau = match self.solve_curvature_quadratic(delta) {
            Some(t) => t,
            None => {
                return false;
            }
        };

        self.curr_zstep += &(tau * &self.curr_dstep);
        true
    }

    pub fn iterate(
        &mut self,
        gradient: &Array1<f64>,
        hessian: &Array2<f64>,
        eps: f64,
        delta: f64,
    ) -> bool {
        if self.curr_iterations >= self.max_iterations {
            return false;
        }

        assert_eq!(gradient.dim(), self.vector_size);
        assert_eq!(hessian.dim(), (self.vector_size, self.vector_size));

        self.curr_zstep.fill(0.0);

        self.curr_rstep = gradient.clone();
        self.curr_dstep = gradient.iter().map(|&x| -x).collect();

        if self.norm(self.curr_rstep.view()) < eps {
            return true;
        }

        for _i in 0..self.vector_size {
            let curvature = self.curr_dstep.t().dot(&hessian.dot(&self.curr_dstep));

            let alpha = (self.curr_rstep.dot(&self.curr_rstep)) / curvature;
            let new_zstep = &self.curr_zstep + alpha * &self.curr_dstep;

            if self.norm(new_zstep.view()) >= delta {
                return self.early_update_zstep(delta);
            }

            let new_rstep = &self.curr_rstep + alpha * &hessian.dot(&self.curr_dstep);
            if self.norm(new_rstep.view()) < eps {
                self.curr_zstep = new_zstep;
                return true;
            }

            let beta = (new_rstep.dot(&new_rstep)) / (self.curr_rstep.dot(&self.curr_rstep));

            self.curr_dstep = beta * &self.curr_dstep - &new_rstep;

            self.curr_zstep = new_zstep;
            self.curr_rstep = new_rstep;
        }

        self.curr_iterations += 1;
        true
    }

    pub fn get_result_readonly(&self) -> ArrayView1<f64> {
        self.curr_zstep.view()
    }

    pub fn get_curr_iterations(&self) -> usize {
        self.curr_iterations
    }

    pub fn get_result(&self) -> Array1<f64> {
        self.curr_zstep.clone()
    }
}
