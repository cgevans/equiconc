//! Library root for the equiconc web app. Both binaries
//! (`equiconc-web`, the main thread, and `equiconc-web-worker`, the
//! solver worker) compile against this crate so they can share wire
//! types, the solver wrapper, and the testcase strings without code
//! duplication or a third intermediary crate.

pub mod app;
pub mod charts;
pub mod exports;
pub mod inputs;
pub mod options;
pub mod outputs;
pub mod solver;
pub mod state;
pub mod testcases;
pub mod wire;
pub mod worker;
pub mod worker_handle;
