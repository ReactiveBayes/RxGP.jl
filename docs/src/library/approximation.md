```@meta
CurrentModule = RxGP
```

# [Approximation methods](@id approximation-reference)

When the input ``x`` to a GP node is itself uncertain (i.e. a distribution rather than a point), kernel expectations of the form ``\mathbb{E}_{q(x)}[k(x, x')]`` must be approximated. RxGP provides approximation methods for this purpose.

## Generalized Unscented Transform

```@docs
GenUnscented
GenUT
GenUnscentedTransform
```

`GenUnscented` (aliases: [`GenUT`](@ref), `GenUnscentedTransform`) extends the classical Unscented Transform by incorporating **skewness** and **kurtosis** of the input distribution to select sigma points and weights. This is particularly useful for non-Gaussian input beliefs.

The sigma points and weights are computed from the first four moments of the distribution:
- Mean ``\mu``
- Covariance ``\Sigma``
- Skewness ``\gamma_1``
- Excess kurtosis ``\gamma_2``

## Kernel expectation approximation

```@docs
approximate_kernel_expectation
approximate_kernel_expectation!
```

These functions approximate expectations of the form:

```math
\bar{g} = \mathbb{E}_{q(x)}\bigl[g(x)\bigr] \approx \sum_{i} w_i \, g(\sigma_i)
```

where ``\sigma_i`` are the sigma/quadrature points and ``w_i`` the corresponding weights.

The in-place variant [`approximate_kernel_expectation!`](@ref) writes the result into a pre-allocated array, avoiding allocations in hot loops.

### Supported approximation methods

| Method | Type | Description |
|:-------|:-----|:------------|
| [`GenUnscented`](@ref) | `GenUnscented()` | Generalized Unscented Transform using skewness and kurtosis |
| Gauss-Hermite | via ReactiveMP's `ghcubature` | Standard cubature-based approximation |

!!! tip
    For uncertain inputs, always set the `method` field in your meta object (e.g. `method = GenUnscented()`). When inputs are observed as `PointMass`, no approximation is needed and the kernel expectations reduce to exact evaluations.
