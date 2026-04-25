//! Baked-in demo testcases. The strings are `include_str!`'d so they
//! ship inside the wasm bundle — no fetch round-trips, no asset paths.

pub struct Testcase {
    pub name: &'static str,
    pub cfe: &'static str,
    pub con: &'static str,
    pub note: &'static str,
}

pub const TESTCASES: &[Testcase] = &[
    Testcase {
        name: "A + B ⇌ AB (1 µM each, ΔG = -10)",
        cfe: include_str!("../testcases/ab_dimer.ocx"),
        con: include_str!("../testcases/ab_dimer.con"),
        note: "Two monomers and one heterodimer.",
    },
    Testcase {
        name: "A homo-oligomerization (monomer/dimer/trimer)",
        cfe: include_str!("../testcases/a_homo.ocx"),
        con: include_str!("../testcases/a_homo.con"),
        note: "One monomer with stoichiometric homo-2 and homo-3 complexes.",
    },
    Testcase {
        name: "ABC competing (3 monomers, 4 complexes)",
        cfe: include_str!("../testcases/abc_competing.ocx"),
        con: include_str!("../testcases/abc_competing.con"),
        note: "Pairwise binders plus an A2B trimer competing for shared monomers.",
    },
];
