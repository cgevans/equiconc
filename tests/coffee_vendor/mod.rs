//! Vendored COFFEE optimizer.
//! Source: https://github.com/coffeesolverdev/coffee
//! See `LICENSE` in this directory for upstream's Apache-2.0 license text.
//!
//! Copyright 2025 coffeesolver.dev (UT Austin Capstone Team FH12 24-25)
//! Licensed under the Apache License, Version 2.0 (the "License");
//! you may not use this file except in compliance with the License.
//! You may obtain a copy of the License at
//!
//!     http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in writing, software
//! distributed under the License is distributed on an "AS IS" BASIS,
//! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
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
