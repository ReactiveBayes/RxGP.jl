# RxGP.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ReactiveBayes.github.io/RxGP.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ReactiveBayes.github.io/RxGP.jl/dev/)
[![Build Status](https://github.com/ReactiveBayes/RxGP.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ReactiveBayes/RxGP.jl/actions/workflows/CI.yml?query=branch%3Amain)

**Sparse Gaussian process factor nodes for reactive message passing on factor graphs.**

RxGP provides dedicated factor nodes and variational message passing update rules for Sparse Variational Gaussian Process (VSGP) models within the [RxInfer.jl](https://github.com/ReactiveBayes/RxInfer.jl) ecosystem. It lets you embed sparse GP computations directly into a Forney-style factor graph so that inference over all latent variables — inducing variables, noise precisions, hyperparameters, and uncertain inputs — is performed automatically.

## Factor Nodes

| Node | Observations | Description |
|:-----|:-------------|:------------|
| `UniSGP` | Scalar `y ∈ ℝ` | Standard univariate VSGP (identity operator) |
| `UniSGP_dID` | Transformed `ỹ ∈ ℝᴾ` | Decoupled inter-domain VSGP with arbitrary linear operators (gradients, joint function+gradient, etc.) |
| `MultiSGP` | Vector `y ∈ ℝᴰ` | Multivariate VSGP via the intrinsic coregionalization model |

## Installation

```julia
using Pkg
Pkg.add("RxGP")
```

Or in development mode:

```julia
Pkg.develop(path="path/to/RxGP.jl")
```

## Quick Start

A minimal GP regression with function-value and gradient observations:

```julia
using RxInfer, RxGP
using KernelFunctions

# --- Data ---
N  = 30
Xu = [[x] for x in range(-4, 4; length=20)]  # inducing inputs
xtrain = [rand() * 8 - 4 for _ in 1:N]       # training inputs
ytrain = sinc.(xtrain) + 0.05 * randn(N)      # noisy observations

# --- Kernel & meta ---
D = 1
mean_fn = (x) -> 0.0
kernel, θ_init, _ = get_simple_kernel_and_params(D; kernel_spec=:SE)
meta = get_UniSGPMeta(D;
    method   = ghcubature(21),
    mean_fn  = mean_fn,
    kernel   = kernel,
    operator = :fn,          # :fn | :grad | :joint_fn_grad
    Xu       = Xu,
    θ        = θ_init,
)

# --- Model ---
@model function gp_regression(y, x, Xu, θ)
    v ~ MvNormalWeightedMeanPrecision(zeros(length(Xu)), 50 * diageye(length(Xu)))
    w ~ GammaShapeRate(1e-2, 1e-2)
    for i in eachindex(y)
        y[i] ~ UniSGP(x[i], v, w, θ)
    end
end

# --- Inference ---
result = infer(
    model       = gp_regression(Xu=Xu, θ=θ_init),
    data        = (y=ytrain, x=[[x] for x in xtrain]),
    meta        = meta,
    iterations  = 10,
    free_energy = true,
)

# --- Predict ---
xtest = [[x] for x in range(-6, 6; length=200)]
means, covs = predict_GP(m_in=xtest, q_v=result.posteriors[:v], q_θ=θ_init, meta=meta)
```

To add **gradient observations**, use `UniSGP_dID` with `operator = :grad` and a Wishart noise prior — see the [full example in the docs](https://ReactiveBayes.github.io/RxGP.jl/stable/examples/usage/).

## Key Features

- **Compose** GP nodes with arbitrary model elements (state-space dynamics, classification likelihoods, etc.) using standard [GraphPPL.jl](https://github.com/ReactiveBayes/GraphPPL.jl) model syntax.
- **Automatic inference** of inducing variables, noise precision, and hyperparameters via variational message passing.
- **Gradient and inter-domain observations** through the `UniSGP_dID` node with configurable linear operators.
- **Flexible kernels** — squared-exponential, multi-SE, spectral-mixture, and combinations — via [KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl).
- **Autodiff and analytic** kernel differentiation (`:AD` / `:AN` modes).

## Documentation

Full documentation is available at **[docs.rxgp.jl](https://ReactiveBayes.github.io/RxGP.jl/stable/)**.

## References

- H.M.H. Nguyen, İ. Şenöz, and B. de Vries, "A Factor Graph Approach to Variational Sparse Gaussian Processes," *IEEE Open Journal of Signal Processing*, 2025.
- A.H. Ledbetter, H.M.H. Nguyen, and B. de Vries, "A Factor-Graph Approach to Decoupled Inter-Domain Variational Sparse Gaussian Processes with Arbitrary Mean Functions," *Under review*, 2026.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file.
