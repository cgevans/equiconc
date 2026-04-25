//! Main-thread entry point. Mounts the Leptos `<App/>` to `<body>`.
//!
//! The actual numerical work runs in a Web Worker spawned from
//! `worker_handle`; this binary stays UI-only.

use equiconc_web::app::App;
use leptos::prelude::*;

fn main() {
    console_error_panic_hook::set_once();
    mount_to_body(App);
}
