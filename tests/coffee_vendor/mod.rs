//! Vendored COFFEE optimizer (Apache-2.0).
//! Source: https://github.com/coffeesolverdev/coffee
//!
//! To replace with a crate dependency, change this to:
//!   pub use coffee::optimize::Optimizer;
//!   pub use coffee::extras::OptimizerArgs;
//! and delete extras.rs, steihaug.rs, optimize.rs, format.rs.

mod extras;
mod format;
mod optimize;
mod steihaug;

pub use extras::OptimizerArgs;
pub use optimize::Optimizer;
