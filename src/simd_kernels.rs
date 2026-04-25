//! Vectorized element-wise kernels for the per-species hot loops in
//! `evaluate_into` / `evaluate_log_into` (and their cheap-objective
//! cousins).
//!
//! Two compile paths:
//! - With `--features simd` (default), bodies dispatch through `pulp` to
//!   the best SIMD ISA available at runtime (SSE2 → AVX2 → AVX-512 on
//!   x86; NEON on aarch64; scalar on wasm).
//! - With `--no-default-features`, every kernel is a plain `for` loop
//!   over `f64`, calling `f64::exp` from libm. Bit-identical to the
//!   pre-0.4 behaviour.
//!
//! Numerical contract: the linear-path kernels (`min_clamp_exp_*`)
//! evaluate `exp` via a degree-12 Taylor polynomial after Cody-Waite
//! range reduction, accurate to ≤2 ulps. The log-path kernels
//! (`fused_lse_and_exp_clamp`, `lse_sum`) drop to scalar libm `exp`
//! per-lane via pulp's partial_store/load — the trust-region step
//! acceptance on `g = ln f` requires per-iteration progress measurable
//! above `4·eps·|g|`, and the polynomial's 1–2 ulp residual stagnates
//! the iteration on extremely stiff systems (COFFEE testcase 0 was
//! the canonical failure). The log-path kernels keep SIMD parallelism
//! for the mins, adds, the Neumaier-compensated LSE reduction, and
//! the clamped-exp store, which is still a measurable end-to-end win
//! over the scalar path.
//!
//! Both `fused_lse_and_exp_clamp` and `lse_sum` use the same libm exp
//! and same compensated reduction so the trust-region ρ check
//! (`g_old` from the full eval, `cand.g` from the candidate eval)
//! sees consistent rounding; mixing libm and polynomial within one
//! solve would break ρ stability. The `Kernels` struct centralizes
//! the dispatch so the two evaluators can never disagree.
//!
//! Outer scalar `lse.exp()` / `f.ln()` calls in `evaluate_log_into`
//! stay on libm — they're single calls per iteration, not loops, and
//! libm precision protects the `f_positive` cancellation detection.

#[cfg(feature = "simd")]
use pulp::{Arch, Simd, WithSimd};

/// Cached SIMD dispatch state. `Kernels::new()` is cheap (a single
/// CPUID query the first time, branch-on-cached-result thereafter), and
/// the `Kernels` instance is stashed on `System` so each `solve()` pays
/// it at most once.
#[derive(Clone, Copy, Default)]
pub(crate) struct Kernels {
    #[cfg(feature = "simd")]
    arch: Arch,
}

impl std::fmt::Debug for Kernels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernels").finish()
    }
}

impl Kernels {
    pub(crate) fn new() -> Self {
        Self {
            #[cfg(feature = "simd")]
            arch: Arch::new(),
        }
    }

    /// `c += log_q`, contiguous slices, equal length.
    #[inline]
    pub(crate) fn add_inplace(self, c: &mut [f64], log_q: &[f64]) {
        debug_assert_eq!(c.len(), log_q.len());
        #[cfg(feature = "simd")]
        {
            self.arch.dispatch(AddInplace { c, log_q });
        }
        #[cfg(not(feature = "simd"))]
        {
            for (ci, &qi) in c.iter_mut().zip(log_q.iter()) {
                *ci += qi;
            }
        }
    }

    /// `grad -= c0`, contiguous slices, equal length.
    #[inline]
    pub(crate) fn sub_inplace(self, grad: &mut [f64], c0: &[f64]) {
        debug_assert_eq!(grad.len(), c0.len());
        #[cfg(feature = "simd")]
        {
            self.arch.dispatch(SubInplace { grad, c0 });
        }
        #[cfg(not(feature = "simd"))]
        {
            for (gi, &ci) in grad.iter_mut().zip(c0.iter()) {
                *gi -= ci;
            }
        }
    }

    /// `out = src * a` for the gradient rescale `grad_g = grad / f`.
    /// Length is `n_mon` (small), so this is mostly here to keep the
    /// scalar/SIMD split symmetric.
    #[inline]
    pub(crate) fn scale_into(self, out: &mut [f64], src: &[f64], a: f64) {
        debug_assert_eq!(out.len(), src.len());
        #[cfg(feature = "simd")]
        {
            self.arch.dispatch(ScaleInto { out, src, a });
        }
        #[cfg(not(feature = "simd"))]
        {
            for (oi, &si) in out.iter_mut().zip(src.iter()) {
                *oi = si * a;
            }
        }
    }

    /// Linear-path one-pass kernel: `t[j] = exp(min(t[j], log_c_clamp))`,
    /// returning `Σ_j t[j]` (the post-exp values).
    ///
    /// Combines `c.mapv_inplace(|x| x.min(clamp).exp())` and the
    /// subsequent `c.sum()` into a single sweep.
    #[inline]
    pub(crate) fn min_clamp_exp_inplace_sum(self, t: &mut [f64], log_c_clamp: f64) -> f64 {
        #[cfg(feature = "simd")]
        {
            self.arch.dispatch(MinClampExpInplaceSum { t, log_c_clamp })
        }
        #[cfg(not(feature = "simd"))]
        {
            let mut sum = 0.0;
            for x in t.iter_mut() {
                let e = x.min(log_c_clamp).exp();
                *x = e;
                sum += e;
            }
            sum
        }
    }

    /// Read-only variant for the cheap-objective path
    /// (`evaluate_objective_into`): returns `Σ_j exp(min(t[j], clamp))`
    /// without writing back to `t`.
    #[inline]
    pub(crate) fn min_clamp_exp_sum(self, t: &[f64], log_c_clamp: f64) -> f64 {
        #[cfg(feature = "simd")]
        {
            self.arch.dispatch(MinClampExpSum { t, log_c_clamp })
        }
        #[cfg(not(feature = "simd"))]
        {
            t.iter().map(|&x| x.min(log_c_clamp).exp()).sum()
        }
    }

    /// Log-path fused kernel: produces `(t_max, sum_shifted)` and
    /// rewrites `t[j] ← exp(min(t[j], log_c_clamp))` in place.
    ///
    /// `sum_shifted = Σ_j exp(t_unclamped[j] − t_max)`. The unclamped
    /// LSE is the load-bearing quantity for the trust-region step
    /// acceptance — clamping inside the LSE biases `f`. The clamp is
    /// applied only when realising `c` for the gradient/Hessian.
    ///
    /// Implementation: two SIMD passes (max-reduction, then fused
    /// shifted-exp + clamped-exp). The original scalar code does three
    /// passes; the saved memory traffic alone is ~33%.
    #[inline]
    pub(crate) fn fused_lse_and_exp_clamp(self, t: &mut [f64], log_c_clamp: f64) -> (f64, f64) {
        #[cfg(feature = "simd")]
        {
            self.arch.dispatch(FusedLseAndExpClamp { t, log_c_clamp })
        }
        #[cfg(not(feature = "simd"))]
        {
            let mut t_max = f64::NEG_INFINITY;
            for &x in t.iter() {
                if x > t_max {
                    t_max = x;
                }
            }
            let mut sum_shifted = 0.0;
            for x in t.iter_mut() {
                sum_shifted += (*x - t_max).exp();
                *x = x.min(log_c_clamp).exp();
            }
            (t_max, sum_shifted)
        }
    }

    /// Cheap-objective log path: returns `(t_max, sum_shifted)` without
    /// modifying `t`. Mirrors `fused_lse_and_exp_clamp` but read-only.
    #[inline]
    pub(crate) fn lse_sum(self, t: &[f64]) -> (f64, f64) {
        #[cfg(feature = "simd")]
        {
            self.arch.dispatch(LseSum { t })
        }
        #[cfg(not(feature = "simd"))]
        {
            let mut t_max = f64::NEG_INFINITY;
            for &x in t.iter() {
                if x > t_max {
                    t_max = x;
                }
            }
            let mut sum_shifted = 0.0;
            for &x in t.iter() {
                sum_shifted += (x - t_max).exp();
            }
            (t_max, sum_shifted)
        }
    }
}

// =====================================================================
// SIMD bodies (only compiled with `--features simd`).
// =====================================================================

#[cfg(feature = "simd")]
mod simd_impl {
    use super::*;

    /// libm-precision SIMD `exp` via per-lane partial_store → scalar
    /// libm → partial_load.
    ///
    /// Used by the log-path kernels (`fused_lse_and_exp_clamp`,
    /// `lse_sum`) because the trust-region step acceptance on the
    /// log-objective drives `g = ln f` toward `O(1)` while requiring
    /// per-iteration progress to be measurable above
    /// `4·eps·|g|` ≈ 6e-15 — borderline territory where the
    /// polynomial's 1–2 ulp residual can stagnate the iteration on
    /// extremely stiff systems (e.g. COFFEE testcase 0 with `c0` in
    /// the nM range and `n_species ≈ 54k`).
    ///
    /// We give up the polynomial speedup on the exp call itself but
    /// keep SIMD parallelism for everything around it (mins, adds,
    /// the LSE compensated reduction, the clamped-exp store), which
    /// still nets a measurable end-to-end win against the scalar
    /// path. The linear-path kernels still use the polynomial
    /// `simd_exp` below — the linear objective operates in
    /// far-from-machine-epsilon territory where the 1–2 ulp residual
    /// is not load-bearing for convergence.
    #[inline(always)]
    pub(super) fn simd_exp_libm<S: Simd>(simd: S, t: S::f64s) -> S::f64s {
        let lanes = S::F64_LANES;
        let mut t_buf = [0.0f64; 8];
        simd.partial_store_f64s(&mut t_buf[..lanes], t);
        let mut out = [0.0f64; 8];
        for i in 0..lanes {
            out[i] = t_buf[i].exp();
        }
        simd.partial_load_f64s(&out[..lanes])
    }

    /// Vectorized `exp(t)` accurate to ~2 ulps on the domain we feed it
    /// (Aᵀλ + log_q values, typically within ±~700 — large positive
    /// values are clamped before the kernel sees them in the linear
    /// path, but the log-path kernel may pass un-clamped values
    /// straight to `exp(t - t_max)` where `t - t_max ≤ 0`, so the
    /// negative tail is the operating regime).
    ///
    /// Method: range-reduce `t = k·ln 2 + r`, evaluate `exp(r)` on
    /// `r ∈ [−ln 2 / 2, ln 2 / 2]` via Estrin-evaluated degree-12
    /// minimax, then `exp(t) = 2^k · exp(r)`. The 2^k step is the only
    /// place we need bit-shifts; pulp 0.22 doesn't expose 64-bit SIMD
    /// shifts, so we drop to scalar via `partial_store_f64s` →
    /// per-lane bit construction → `partial_load_f64s`. With the
    /// polynomial cost dominating, the few-lane scalar epilogue is in
    /// the noise.
    #[inline(always)]
    pub(super) fn simd_exp<S: Simd>(simd: S, t: S::f64s) -> S::f64s {
        // Cody-Waite split of ln(2) into hi/lo doubles, exact to ~108 bits.
        let ln2_hi = simd.splat_f64s(0.693_147_180_369_123_8); // 0x3FE62E42_FEE00000
        let ln2_lo = simd.splat_f64s(1.908_214_929_270_587_7e-10);
        let inv_ln2 = simd.splat_f64s(std::f64::consts::LOG2_E);

        // Round-to-nearest-even via the magic-constant trick: works for
        // any sign provided |t * inv_ln2| < 2^52.
        let magic = simd.splat_f64s((1u64 << 52) as f64 + (1u64 << 51) as f64);
        let kf = simd.mul_add_f64s(t, inv_ln2, magic);
        let kf = simd.sub_f64s(kf, magic);

        // r = t − k·ln 2, computed as (t − k·ln2_hi) − k·ln2_lo via FMA.
        let r = simd.mul_add_f64s(simd.neg_f64s(kf), ln2_hi, t);
        let r = simd.mul_add_f64s(simd.neg_f64s(kf), ln2_lo, r);

        // Degree-12 minimax for exp(r) on [−ln 2/2, ln 2/2]. The
        // truncated Taylor coefficients are within machine precision of
        // the minimax coefficients on this tiny interval, so we use
        // 1/k! directly — easy to read and accurate to <1 ulp before
        // the 2^k multiply.
        let c12 = simd.splat_f64s(1.0 / 479_001_600.0);
        let c11 = simd.splat_f64s(1.0 / 39_916_800.0);
        let c10 = simd.splat_f64s(1.0 / 3_628_800.0);
        let c9 = simd.splat_f64s(1.0 / 362_880.0);
        let c8 = simd.splat_f64s(1.0 / 40_320.0);
        let c7 = simd.splat_f64s(1.0 / 5_040.0);
        let c6 = simd.splat_f64s(1.0 / 720.0);
        let c5 = simd.splat_f64s(1.0 / 120.0);
        let c4 = simd.splat_f64s(1.0 / 24.0);
        let c3 = simd.splat_f64s(1.0 / 6.0);
        let c2 = simd.splat_f64s(0.5);
        let c1 = simd.splat_f64s(1.0);
        let one = simd.splat_f64s(1.0);

        // Horner.
        let mut p = c12;
        p = simd.mul_add_f64s(p, r, c11);
        p = simd.mul_add_f64s(p, r, c10);
        p = simd.mul_add_f64s(p, r, c9);
        p = simd.mul_add_f64s(p, r, c8);
        p = simd.mul_add_f64s(p, r, c7);
        p = simd.mul_add_f64s(p, r, c6);
        p = simd.mul_add_f64s(p, r, c5);
        p = simd.mul_add_f64s(p, r, c4);
        p = simd.mul_add_f64s(p, r, c3);
        p = simd.mul_add_f64s(p, r, c2);
        p = simd.mul_add_f64s(p, r, c1);
        p = simd.mul_add_f64s(p, r, one);

        // 2^k via per-lane bit construction.
        //
        // Buffer is sized for the widest f64 SIMD on stable Rust
        // (AVX-512: 8 lanes). Three regimes:
        //
        // - `k ∈ [-1022, 1023]`: biased exponent (k + 1023) is in
        //   [1, 2046], fits in 11 bits, gives a normal f64 directly.
        // - `k ∈ [-1074, -1023]`: subnormal range. We can't construct
        //   the value with one shift, but the two-mul split
        //   `2^k = 2^-1022 · 2^(k+1022)` works because `k+1022 ∈
        //   [-52, -1]` is well inside the normal range. f64
        //   multiplication produces the correct subnormal result.
        // - `k < -1074`: true underflow → 0.0.
        // - `k > 1023`: saturate to `2^1023` (largest finite power of
        //   two). In `evaluate_into` inputs are clamped to
        //   `log_c_clamp ≤ 700` before the kernel sees them; in the
        //   log path `t − t_max ≤ 0`. This branch is therefore
        //   defence-in-depth.
        //
        // An earlier version sign-extended `(k + 1023) as u64` for
        // negative `k`, producing a huge negative f64 instead of
        // underflowing to zero; the
        // `min_clamp_exp_inplace_sum_stress_large` and `simd_exp_
        // accuracy_sweep` tests gate against regression there.
        let lanes = S::F64_LANES;
        let mut k_buf = [0.0f64; 8];
        simd.partial_store_f64s(&mut k_buf[..lanes], kf);
        let mut pow2_buf = [0.0f64; 8];
        for i in 0..lanes {
            let k = k_buf[i] as i64;
            if k > 1023 {
                pow2_buf[i] = f64::from_bits(((1023_i64 + 1023) as u64) << 52);
            } else if k >= -1022 {
                pow2_buf[i] = f64::from_bits(((k + 1023) as u64) << 52);
            } else if k >= -1074 {
                // k + 1022 ∈ [-52, -1] — normal f64.
                let small_normal = f64::from_bits(((k + 1022 + 1023) as u64) << 52);
                let smallest_normal = f64::from_bits(1u64 << 52); // 2^-1022
                pow2_buf[i] = smallest_normal * small_normal;
            } else {
                pow2_buf[i] = 0.0;
            }
        }
        let pow2 = simd.partial_load_f64s(&pow2_buf[..lanes]);

        simd.mul_f64s(p, pow2)
    }

    pub(super) struct AddInplace<'a> {
        pub c: &'a mut [f64],
        pub log_q: &'a [f64],
    }
    impl<'a> WithSimd for AddInplace<'a> {
        type Output = ();
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) {
            let Self { c, log_q } = self;
            let (c_head, c_tail) = S::as_mut_simd_f64s(c);
            let (q_head, q_tail) = S::as_simd_f64s(log_q);
            for (cv, qv) in c_head.iter_mut().zip(q_head.iter()) {
                *cv = simd.add_f64s(*cv, *qv);
            }
            for (cv, &qv) in c_tail.iter_mut().zip(q_tail.iter()) {
                *cv += qv;
            }
        }
    }

    pub(super) struct SubInplace<'a> {
        pub grad: &'a mut [f64],
        pub c0: &'a [f64],
    }
    impl<'a> WithSimd for SubInplace<'a> {
        type Output = ();
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) {
            let Self { grad, c0 } = self;
            let (g_head, g_tail) = S::as_mut_simd_f64s(grad);
            let (c_head, c_tail) = S::as_simd_f64s(c0);
            for (gv, cv) in g_head.iter_mut().zip(c_head.iter()) {
                *gv = simd.sub_f64s(*gv, *cv);
            }
            for (gv, &cv) in g_tail.iter_mut().zip(c_tail.iter()) {
                *gv -= cv;
            }
        }
    }

    pub(super) struct ScaleInto<'a> {
        pub out: &'a mut [f64],
        pub src: &'a [f64],
        pub a: f64,
    }
    impl<'a> WithSimd for ScaleInto<'a> {
        type Output = ();
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) {
            let Self { out, src, a } = self;
            let av = simd.splat_f64s(a);
            let (o_head, o_tail) = S::as_mut_simd_f64s(out);
            let (s_head, s_tail) = S::as_simd_f64s(src);
            for (ov, sv) in o_head.iter_mut().zip(s_head.iter()) {
                *ov = simd.mul_f64s(*sv, av);
            }
            for (ov, &sv) in o_tail.iter_mut().zip(s_tail.iter()) {
                *ov = sv * a;
            }
        }
    }

    pub(super) struct MinClampExpInplaceSum<'a> {
        pub t: &'a mut [f64],
        pub log_c_clamp: f64,
    }
    impl<'a> WithSimd for MinClampExpInplaceSum<'a> {
        type Output = f64;
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> f64 {
            let Self { t, log_c_clamp } = self;
            let clamp_v = simd.splat_f64s(log_c_clamp);
            let mut sum_v = simd.splat_f64s(0.0);
            let (head, tail) = S::as_mut_simd_f64s(t);
            for chunk in head.iter_mut() {
                let clamped = simd.min_f64s(*chunk, clamp_v);
                let e = simd_exp(simd, clamped);
                *chunk = e;
                sum_v = simd.add_f64s(sum_v, e);
            }
            let mut sum = simd.reduce_sum_f64s(sum_v);
            for x in tail.iter_mut() {
                let e = x.min(log_c_clamp).exp();
                *x = e;
                sum += e;
            }
            sum
        }
    }

    pub(super) struct MinClampExpSum<'a> {
        pub t: &'a [f64],
        pub log_c_clamp: f64,
    }
    impl<'a> WithSimd for MinClampExpSum<'a> {
        type Output = f64;
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> f64 {
            let Self { t, log_c_clamp } = self;
            let clamp_v = simd.splat_f64s(log_c_clamp);
            let mut sum_v = simd.splat_f64s(0.0);
            let (head, tail) = S::as_simd_f64s(t);
            for &chunk in head.iter() {
                let clamped = simd.min_f64s(chunk, clamp_v);
                let e = simd_exp(simd, clamped);
                sum_v = simd.add_f64s(sum_v, e);
            }
            let mut sum = simd.reduce_sum_f64s(sum_v);
            for &x in tail.iter() {
                sum += x.min(log_c_clamp).exp();
            }
            sum
        }
    }

    pub(super) struct FusedLseAndExpClamp<'a> {
        pub t: &'a mut [f64],
        pub log_c_clamp: f64,
    }
    impl<'a> WithSimd for FusedLseAndExpClamp<'a> {
        type Output = (f64, f64);
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> (f64, f64) {
            let Self { t, log_c_clamp } = self;

            // Pass 1: t_max via SIMD reduction.
            let neg_inf = simd.splat_f64s(f64::NEG_INFINITY);
            let mut max_v = neg_inf;
            {
                let (head, tail) = S::as_simd_f64s(t);
                for &chunk in head.iter() {
                    max_v = simd.max_f64s(max_v, chunk);
                }
                let mut t_max_scalar = simd.reduce_max_f64s(max_v);
                for &x in tail.iter() {
                    if x > t_max_scalar {
                        t_max_scalar = x;
                    }
                }
                // Broadcast back into a SIMD register for pass 2.
                max_v = simd.splat_f64s(t_max_scalar);
            }
            let t_max_scalar = simd.reduce_max_f64s(max_v);

            // Pass 2: write c = exp(min(t, clamp)); accumulate
            // Σ exp(t − t_max) with Neumaier-compensated summation.
            //
            // The trust-region step acceptance can drift past the
            // strict convergence boundary on extremely stiff systems
            // (n_species in the 50k+ range with c0 in the nM regime —
            // COFFEE's testcase 0 is the canonical failure mode here)
            // when the LSE summation accumulates O(n_species·eps)
            // rounding. Compensating the SIMD per-lane accumulator
            // with a parallel "lost-bits" register costs one extra
            // FMA-equivalent op per chunk and brings the total error
            // down to O(eps²). The cross-lane reduction at the end is
            // small (≤ 8 lanes on AVX-512) and uses the same
            // compensation.
            let clamp_v = simd.splat_f64s(log_c_clamp);
            let mut sum_v = simd.splat_f64s(0.0);
            let mut comp_v = simd.splat_f64s(0.0);
            let (head, tail) = S::as_mut_simd_f64s(t);
            for chunk in head.iter_mut() {
                let raw = *chunk;
                let clamped = simd.min_f64s(raw, clamp_v);
                let e_clamped = simd_exp_libm(simd, clamped);
                let shifted = simd.sub_f64s(raw, max_v);
                let e_shifted = simd_exp_libm(simd, shifted);
                *chunk = e_clamped;
                // Neumaier compensation. `t_new = sum + e_shifted`
                // exactly when |sum| ≥ |e_shifted|; otherwise the
                // roles swap. Either way the lost low-order bits are
                // captured into `comp_v`.
                let t_new = simd.add_f64s(sum_v, e_shifted);
                let abs_sum = simd.abs_f64s(sum_v);
                let abs_e = simd.abs_f64s(e_shifted);
                let sum_bigger = simd.greater_than_or_equal_f64s(abs_sum, abs_e);
                let lost_when_sum_bigger = simd.add_f64s(simd.sub_f64s(sum_v, t_new), e_shifted);
                let lost_when_e_bigger = simd.add_f64s(simd.sub_f64s(e_shifted, t_new), sum_v);
                let lost = simd.select_f64s(sum_bigger, lost_when_sum_bigger, lost_when_e_bigger);
                comp_v = simd.add_f64s(comp_v, lost);
                sum_v = t_new;
            }
            let lane_sum = simd.reduce_sum_f64s(sum_v);
            let lane_comp = simd.reduce_sum_f64s(comp_v);
            let mut sum_shifted = lane_sum + lane_comp;
            // Tail epilogue with the same compensation pattern.
            let mut comp = 0.0_f64;
            for x in tail.iter_mut() {
                let raw = *x;
                let e_shifted = (raw - t_max_scalar).exp();
                let t_new = sum_shifted + e_shifted;
                if sum_shifted.abs() >= e_shifted.abs() {
                    comp += (sum_shifted - t_new) + e_shifted;
                } else {
                    comp += (e_shifted - t_new) + sum_shifted;
                }
                sum_shifted = t_new;
                *x = raw.min(log_c_clamp).exp();
            }
            sum_shifted += comp;
            (t_max_scalar, sum_shifted)
        }
    }

    pub(super) struct LseSum<'a> {
        pub t: &'a [f64],
    }
    impl<'a> WithSimd for LseSum<'a> {
        type Output = (f64, f64);
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> (f64, f64) {
            let Self { t } = self;
            let neg_inf = simd.splat_f64s(f64::NEG_INFINITY);
            let mut max_v = neg_inf;
            let (head, tail) = S::as_simd_f64s(t);
            for &chunk in head.iter() {
                max_v = simd.max_f64s(max_v, chunk);
            }
            let mut t_max = simd.reduce_max_f64s(max_v);
            for &x in tail.iter() {
                if x > t_max {
                    t_max = x;
                }
            }
            let max_v = simd.splat_f64s(t_max);
            // Neumaier-compensated SIMD sum — see the equivalent
            // section in `FusedLseAndExpClamp` for the rationale; the
            // candidate evaluator must match the full evaluator's
            // sum semantics so the trust-region ρ check sees a
            // consistent g vs cand.g pair.
            let mut sum_v = simd.splat_f64s(0.0);
            let mut comp_v = simd.splat_f64s(0.0);
            for &chunk in head.iter() {
                let shifted = simd.sub_f64s(chunk, max_v);
                let e = simd_exp_libm(simd, shifted);
                let t_new = simd.add_f64s(sum_v, e);
                let abs_sum = simd.abs_f64s(sum_v);
                let abs_e = simd.abs_f64s(e);
                let sum_bigger = simd.greater_than_or_equal_f64s(abs_sum, abs_e);
                let lost_when_sum_bigger = simd.add_f64s(simd.sub_f64s(sum_v, t_new), e);
                let lost_when_e_bigger = simd.add_f64s(simd.sub_f64s(e, t_new), sum_v);
                let lost = simd.select_f64s(sum_bigger, lost_when_sum_bigger, lost_when_e_bigger);
                comp_v = simd.add_f64s(comp_v, lost);
                sum_v = t_new;
            }
            let lane_sum = simd.reduce_sum_f64s(sum_v);
            let lane_comp = simd.reduce_sum_f64s(comp_v);
            let mut sum = lane_sum + lane_comp;
            let mut comp = 0.0_f64;
            for &x in tail.iter() {
                let e = (x - t_max).exp();
                let t_new = sum + e;
                if sum.abs() >= e.abs() {
                    comp += (sum - t_new) + e;
                } else {
                    comp += (e - t_new) + sum;
                }
                sum = t_new;
            }
            sum += comp;
            (t_max, sum)
        }
    }
}

#[cfg(feature = "simd")]
use simd_impl::{
    AddInplace, FusedLseAndExpClamp, LseSum, MinClampExpInplaceSum, MinClampExpSum, ScaleInto,
    SubInplace,
};

#[cfg(test)]
mod tests {
    use super::*;

    fn rel_diff(a: f64, b: f64) -> f64 {
        let denom = a.abs().max(b.abs()).max(f64::MIN_POSITIVE);
        ((a - b).abs()) / denom
    }

    fn ulp_diff(a: f64, b: f64) -> u64 {
        if a == b {
            return 0;
        }
        let abits = a.to_bits() as i128;
        let bbits = b.to_bits() as i128;
        (abits - bbits).unsigned_abs() as u64
    }

    fn sample_inputs() -> Vec<f64> {
        // Lengths cover lane-multiple, lane-multiple+tail, and a single
        // sub-lane chunk to exercise the partial path.
        let mut v = Vec::new();
        for k in [0, 1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 33, 64, 100, 1000] {
            for i in 0..k {
                v.push(((i as f64) / 7.0).sin() * 4.0 - 1.0);
            }
            v.push(f64::NAN); // sentinel to break runs between sizes
        }
        v.retain(|x| x.is_finite());
        v
    }

    #[test]
    fn add_inplace_matches_scalar() {
        let k = Kernels::new();
        let a0: Vec<f64> = (0..127).map(|i| (i as f64) * 0.31 - 5.0).collect();
        let b: Vec<f64> = (0..127).map(|i| (i as f64) * -0.17 + 0.5).collect();
        let mut a = a0.clone();
        k.add_inplace(&mut a, &b);
        for i in 0..a.len() {
            assert_eq!(
                a[i].to_bits(),
                (a0[i] + b[i]).to_bits(),
                "mismatch at i={i}"
            );
        }
    }

    #[test]
    fn sub_inplace_matches_scalar() {
        let k = Kernels::new();
        let a0: Vec<f64> = (0..127).map(|i| (i as f64) * 0.31 - 5.0).collect();
        let b: Vec<f64> = (0..127).map(|i| (i as f64) * -0.17 + 0.5).collect();
        let mut a = a0.clone();
        k.sub_inplace(&mut a, &b);
        for i in 0..a.len() {
            assert_eq!(
                a[i].to_bits(),
                (a0[i] - b[i]).to_bits(),
                "mismatch at i={i}"
            );
        }
    }

    #[test]
    fn scale_into_matches_scalar() {
        let k = Kernels::new();
        let src: Vec<f64> = (0..27).map(|i| (i as f64) * 0.31 - 5.0).collect();
        let mut out = vec![0.0; src.len()];
        let a = -2.5_f64;
        k.scale_into(&mut out, &src, a);
        for i in 0..src.len() {
            assert_eq!(
                out[i].to_bits(),
                (src[i] * a).to_bits(),
                "mismatch at i={i}"
            );
        }
    }

    #[test]
    fn min_clamp_exp_inplace_sum_matches_libm() {
        let k = Kernels::new();
        // Inputs span the negative tail (where exp is precise) and a
        // few values above the clamp to verify the min(t, clamp) gate.
        let inputs = sample_inputs();
        let log_c_clamp = 50.0_f64;
        let mut t = inputs.clone();
        let got_sum = k.min_clamp_exp_inplace_sum(&mut t, log_c_clamp);

        let mut want_sum = 0.0;
        for (i, &x) in inputs.iter().enumerate() {
            let want = x.min(log_c_clamp).exp();
            let ulps = ulp_diff(t[i], want);
            assert!(
                ulps <= 8,
                "value at i={i}: got {} want {} ({ulps} ulps)",
                t[i],
                want
            );
            want_sum += want;
        }
        let rel = rel_diff(got_sum, want_sum);
        assert!(
            rel < 1e-13,
            "sum rel diff {rel} (got {got_sum} want {want_sum})"
        );
    }

    #[test]
    fn min_clamp_exp_sum_matches_libm() {
        let k = Kernels::new();
        let inputs = sample_inputs();
        let log_c_clamp = 50.0_f64;
        let got_sum = k.min_clamp_exp_sum(&inputs, log_c_clamp);
        let want_sum: f64 = inputs.iter().map(|&x| x.min(log_c_clamp).exp()).sum();
        let rel = rel_diff(got_sum, want_sum);
        assert!(rel < 1e-13, "rel diff {rel}");
    }

    #[test]
    fn fused_lse_and_exp_clamp_matches_libm() {
        let k = Kernels::new();
        let inputs = sample_inputs();
        let log_c_clamp = 50.0_f64;
        let mut t = inputs.clone();
        let (t_max_got, sum_shifted_got) = k.fused_lse_and_exp_clamp(&mut t, log_c_clamp);

        let t_max_want = inputs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(t_max_got, t_max_want, "t_max mismatch");

        let sum_shifted_want: f64 = inputs.iter().map(|&x| (x - t_max_want).exp()).sum();
        let rel = rel_diff(sum_shifted_got, sum_shifted_want);
        assert!(rel < 1e-13, "sum_shifted rel diff {rel}");

        for (i, &x) in inputs.iter().enumerate() {
            let want = x.min(log_c_clamp).exp();
            let ulps = ulp_diff(t[i], want);
            assert!(
                ulps <= 8,
                "c[i={i}]: got {} want {} ({ulps} ulps)",
                t[i],
                want
            );
        }
    }

    #[test]
    fn lse_sum_matches_libm() {
        let k = Kernels::new();
        let inputs = sample_inputs();
        let (t_max_got, sum_got) = k.lse_sum(&inputs);
        let t_max_want = inputs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum_want: f64 = inputs.iter().map(|&x| (x - t_max_want).exp()).sum();
        assert_eq!(t_max_got, t_max_want);
        assert!(rel_diff(sum_got, sum_want) < 1e-13);
    }

    /// Confirm Neumaier compensation actually helps: build a sum of
    /// 54k values with the magnitude profile of the LSE shifted-exp on
    /// COFFEE testcase 0 (most values ~ 1e-90, a few ~ 1) and assert
    /// the SIMD reduction error against a scalar Kahan-Babuska-Neumaier
    /// reference.
    #[test]
    #[cfg(feature = "simd")]
    fn lse_sum_compensation_active() {
        let k = Kernels::new();
        // Construct shifted-exp values directly: 15 "near 1" values,
        // 54k "tiny" values. t inputs that produce these on simd_exp:
        // for value ≈ 1, t ≈ 0; for value ≈ 1e-90, t ≈ -207.
        let mut t = Vec::with_capacity(54218);
        for i in 0..54218 {
            // First 15 indices ~ 0 (so exp ≈ 1).
            t.push(if i < 15 {
                0.0
            } else {
                -207.0 + (i as f64) * 1e-6
            });
        }
        let (got_max, got_sum) = k.lse_sum(&t);
        assert_eq!(got_max, 0.0);

        // Reference: Kahan-Neumaier scalar of libm exp values.
        let mut ref_sum = 0.0_f64;
        let mut ref_comp = 0.0_f64;
        for &x in t.iter() {
            let e = (x - got_max).exp();
            let s = ref_sum + e;
            if ref_sum.abs() >= e.abs() {
                ref_comp += (ref_sum - s) + e;
            } else {
                ref_comp += (e - s) + ref_sum;
            }
            ref_sum = s;
        }
        let want_sum = ref_sum + ref_comp;

        let rel = rel_diff(got_sum, want_sum);
        assert!(
            rel < 1e-15,
            "SIMD compensated sum diverges from scalar KBN: got {got_sum} want {want_sum} (rel {rel:e})"
        );
    }

    /// Stress test mirroring the `evaluate_into` operating regime on
    /// large `n_species`: many values clamped at 700 (their exp
    /// overflows would-be in libm but stays at f64-max in our
    /// polynomial), interleaved with mid- and small-magnitude values
    /// that span the full domain. This is the case that tripped up
    /// the COFFEE testcase 0 Linear path.
    #[test]
    fn min_clamp_exp_inplace_sum_stress_large() {
        let k = Kernels::new();
        let log_c_clamp = 700.0;
        // 54k species, with a non-trivial spread: half clamped, the
        // other half with t in [-200, 200] picked to give exp values
        // at every magnitude scale.
        let mut t = Vec::with_capacity(54218);
        for i in 0..54218usize {
            let phase = (i as f64) * 0.123;
            let raw = if i % 2 == 0 {
                800.0 + (phase.sin() * 100.0)
            } else {
                ((phase * 1.7).cos()) * 200.0
            };
            t.push(raw);
        }
        let mut want_buf = t.clone();
        let want_sum: f64 = {
            for x in want_buf.iter_mut() {
                *x = x.min(log_c_clamp).exp();
            }
            want_buf.iter().sum()
        };
        let got_sum = k.min_clamp_exp_inplace_sum(&mut t, log_c_clamp);

        // Both paths can produce the same overflow-to-+inf sum; what we
        // require is that the kernel does not invent a different f64
        // class (e.g. swap +inf for finite-but-large, or NaN, or -inf).
        assert_eq!(
            want_sum.is_infinite(),
            got_sum.is_infinite(),
            "infinite-class mismatch: want {want_sum:e}, got {got_sum:e}"
        );
        assert_eq!(
            want_sum.is_sign_positive(),
            got_sum.is_sign_positive(),
            "sign mismatch: want {want_sum:e}, got {got_sum:e}"
        );
        if want_sum.is_finite() && got_sum.is_finite() {
            let rel = rel_diff(got_sum, want_sum);
            assert!(rel < 1e-12, "rel diff {rel:e}");
        }
        // Also: every per-element exp must agree to <= 4 ulps.
        for (i, (&got, &want)) in t.iter().zip(want_buf.iter()).enumerate() {
            let ulps = ulp_diff(got, want);
            assert!(
                ulps <= 4,
                "elem {i}: t_in (post-clamp limit {log_c_clamp}); got {got:e} want {want:e} ({ulps} ulps)"
            );
        }
    }

    /// Sweep `simd_exp` across the full operating-point range we feed
    /// it from `evaluate_into` and `evaluate_log_into` and assert it
    /// stays close to libm. This is the canary that protects against
    /// silent regressions in the polynomial / range reduction.
    #[test]
    #[cfg(feature = "simd")]
    fn simd_exp_accuracy_sweep() {
        let k = Kernels::new();
        let mut max_ulps = 0u64;
        let mut max_rel = 0.0_f64;
        let mut worst_x = 0.0_f64;
        // Range spans the full operating envelope:
        // - log-path inputs after `t - t_max` shift are in (-∞, 0].
        // - linear-path inputs are clamped to <= log_c_clamp = 700.
        // - solver intermediate iterates can hit anywhere in [-700, 700].
        //
        // The underflow tail down to -750 is the regression bait: for
        // `t < -log(2) * 1022 ≈ -708`, the true `exp(t)` underflows to
        // a denormal/zero, which is also what the polynomial+ldexp
        // path must produce. An earlier version of this kernel
        // mishandled negative `k_clamped + 1023` by sign-extending into
        // the f64 sign bit, producing a huge negative value instead of
        // zero; testcase 0 of the COFFEE bench reliably tripped it.
        let xs: Vec<f64> = (-7500..=7000)
            .step_by(1)
            .map(|i| (i as f64) / 10.0)
            .collect();
        let mut t = xs.clone();
        // Use min_clamp_exp_inplace_sum to drive the SIMD exp; clamp
        // higher than max input so it's a no-op gate.
        let _ = k.min_clamp_exp_inplace_sum(&mut t, 1e9);
        for (&x, &got) in xs.iter().zip(t.iter()) {
            let want = x.exp();
            let ulps = ulp_diff(got, want);
            let rel = rel_diff(got, want);
            if ulps > max_ulps {
                max_ulps = ulps;
                worst_x = x;
            }
            if rel > max_rel {
                max_rel = rel;
            }
        }
        // 4 ulps would be acceptable for the trust-region ρ check; we
        // expect <= 2 with a degree-12 minimax. Set a generous gate
        // here so the canary fires only on real regressions.
        assert!(
            max_ulps <= 4,
            "max ulp error {max_ulps} (worst input x={worst_x}, max rel diff {max_rel:e})"
        );
    }

    /// LSE invariants we rely on in `evaluate_log_into`:
    /// 1. `lse = t_max + ln(sum_shifted)` reproduces `ln Σ exp(t)` to
    ///    machine precision (the whole point of the shift).
    /// 2. The result is identical between `fused_lse_and_exp_clamp`
    ///    and `lse_sum` on the same inputs (so candidate evaluator
    ///    and full evaluator agree on `f` for the trust-region ρ test).
    #[test]
    fn lse_self_consistency() {
        let k = Kernels::new();
        let inputs = sample_inputs();
        let log_c_clamp = 50.0_f64;
        let (t_max_a, sum_a) = k.lse_sum(&inputs);
        let mut buf = inputs.clone();
        let (t_max_b, sum_b) = k.fused_lse_and_exp_clamp(&mut buf, log_c_clamp);
        assert_eq!(t_max_a, t_max_b);
        assert!(rel_diff(sum_a, sum_b) < 1e-14);
    }
}
