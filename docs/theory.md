# Implementation notes

**The below text is LLM-generated, based on the implementation. It has not
yet been reviewed and edited, and should not be trusted without checking
for correctness.**

equiconc solves for equilibrium concentrations using the convex optimization
approach of [Dirks et al. (2007), *SIAM Review* 49(1), 65--88](https://doi.org/10.1137/060651100).
This page summarizes the mathematical formulation.

## Problem setup

Consider a system with \(M\) monomer species and \(J\) total species (monomers +
complexes). Each species \(j\) has a concentration \(c_j\), and each complex is
defined by its stoichiometry: how many copies of each monomer it contains.

The **stoichiometry matrix** \(\mathbf{A} \in \mathbb{R}^{M \times J}\) has
entry \(A_{ij}\) equal to the number of copies of monomer \(i\) in species \(j\).
For monomer species, \(\mathbf{A}\) contains the \(M \times M\) identity matrix.

## Equilibrium conditions

At equilibrium, concentrations satisfy two constraints:

**Mass conservation.** The total amount of each monomer is conserved:

\[
\sum_{j=1}^{J} A_{ij}\, c_j = c_i^0 \quad \text{for } i = 1, \ldots, M
\]

where \(c_i^0\) is the total (initial) concentration of monomer \(i\).

**Chemical equilibrium.** Each complex concentration is related to the free
monomer concentrations by the partition function:

\[
c_j = Q_j \prod_{i=1}^{M} \left(\tilde{c}_i\right)^{A_{ij}}
\]

where \(\tilde{c}_i\) is the free concentration of monomer \(i\), and

\[
Q_j = \exp\!\left(-\frac{\Delta G_j^\circ}{RT}\right)
\]

is the Boltzmann factor for species \(j\) (with \(Q_i = 1\) for monomers,
i.e., \(\Delta G^\circ = 0\)).

## The dual problem

Rather than solving the primal (nonlinear mass conservation equations) directly,
equiconc minimizes the **convex dual** objective:

\[
f(\boldsymbol{\lambda}) = -\boldsymbol{\lambda}^\top \mathbf{c}^0
+ \sum_{j=1}^{J} Q_j \exp\!\left(\sum_{i=1}^{M} A_{ij}\, \lambda_i\right)
\]

where \(\boldsymbol{\lambda} \in \mathbb{R}^M\) are the dual variables
(one per monomer species). The key insight is that the dimension of this
optimization equals the number of monomer species (typically 2--10),
making it fast regardless of how many complexes exist.

### Gradient and Hessian

The gradient is:

\[
\nabla f(\boldsymbol{\lambda}) = -\mathbf{c}^0 + \mathbf{A}\, \mathbf{c}(\boldsymbol{\lambda})
\]

where \(c_j(\boldsymbol{\lambda}) = Q_j \exp(\mathbf{A}^\top \boldsymbol{\lambda})_j\).
At the optimum, \(\nabla f = 0\) recovers the mass conservation equations.

The Hessian is:

\[
\mathbf{H} = \mathbf{A}\, \text{diag}(\mathbf{c})\, \mathbf{A}^\top
\]

Since all \(c_j > 0\), the Hessian is positive definite, guaranteeing that
\(f\) is strictly convex and has a unique minimum.

## Trust-region Newton method

equiconc uses a **trust-region Newton method** with **dog-leg steps**:

1. At each iteration, compute the Newton step
   \(\mathbf{p}_N = -\mathbf{H}^{-1} \nabla f\) via Cholesky decomposition
   (always succeeds since \(\mathbf{H} \succ 0\)).

2. If \(\|\mathbf{p}_N\| \leq \Delta\) (trust-region radius), take the full
   Newton step.

3. Otherwise, compute the Cauchy (steepest descent) step and interpolate along
   the dog-leg path to stay within the trust region.

4. Update the trust-region radius based on the ratio of actual to predicted
   reduction.

### Log-space evaluation

To prevent floating-point overflow when concentrations span many orders of
magnitude, the objective, gradient, and Hessian are evaluated in log space:

\[
\log c_j = \log Q_j + \sum_{i} A_{ij}\, \lambda_i
\]

Values of \(\log c_j\) are clamped to prevent \(\exp(\cdot)\) from exceeding
`f64::MAX`.

## Recovering primal concentrations

Once the dual optimum \(\boldsymbol{\lambda}^*\) is found, primal
concentrations are recovered as:

\[
c_j^* = \exp\!\left(\log Q_j + (\mathbf{A}^\top \boldsymbol{\lambda}^*)_j\right)
\]

## References

- Dirks, R. M., Bois, J. S., Schaeffer, J. M., Winfree, E., & Pierce, N. A.
  (2007). Thermodynamic analysis of interacting nucleic acid strands.
  *SIAM Review*, 49(1), 65--88.
  [doi:10.1137/060651100](https://doi.org/10.1137/060651100)
