```@meta
CurrentModule = RxGP
```

# RxGP.jl

*Julia package for sparse Gaussian process inference via reactive message passing on factor graphs.*

RxGP provides custom factor nodes and message passing update rules for Sparse Variational Gaussian Process (SVGP) models within the [RxInfer.jl](https://github.com/ReactiveBayes/RxInfer.jl) ecosystem. It enables efficient, scalable GP regression and GP state-space modelling by embedding sparse GP computations directly into a factor graph, where inference is performed automatically through message passing.

## Why RxGP?

Gaussian processes are powerful non-parametric models, but standard GP inference scales cubically in the number of observations. Sparse variational approximations reduce this cost by introducing a small set of **inducing points**, but integrating sparse GPs into larger probabilistic models (e.g. state-space models) typically requires custom, hand-derived inference algorithms.

RxGP solves this by packaging the sparse GP computation into dedicated **factor nodes** — [`UniSGP`](@ref), [`UniSGP_dID`](@ref), and [`MultiSGP`](@ref) — that plug directly into the [ReactiveMP.jl](https://github.com/ReactiveBayes/ReactiveMP.jl) message passing engine. This means you can:

- **Compose** GP components with arbitrary probabilistic model elements (state-space dynamics, switching models, etc.) using standard [GraphPPL.jl](https://github.com/ReactiveBayes/GraphPPL.jl) syntax.
- **Infer** all latent variables (function values, inducing variables, noise precisions, hyperparameters, and uncertain inputs) jointly via variational message passing — no manual derivations required.
- **Observe** function values, gradients, or both simultaneously with the `:fn`, `:grad`, and `:joint_fn_grad` operator modes.

## Package Features

- **Three sparse GP factor nodes** with full message passing rules on every edge:
  - [`UniSGP`](@ref) — Univariate (scalar output) sparse GP observations.
  - [`UniSGP_dID`](@ref) — Univariate sparse GP with derivative/gradient observations.
  - [`MultiSGP`](@ref) — Multivariate (vector output) sparse GP observations.
- **Flexible kernel specification** via [`get_simple_kernel_and_params`](@ref), supporting squared-exponential (`:SE`), multi-SE (`:SEn`), spectral-mixture (`:SMn`), and combined (`:SEn_SMn`) kernels.
- **Automatic and analytic differentiation** of kernels for gradient observation models (`:AD` and `:AN` modes).
- **Generalized Unscented Transform** ([`GenUnscented`](@ref)) for approximate kernel expectations under uncertain inputs.
- **Posterior prediction** via [`predict_GP`](@ref) for test-time inference with trained models.
- **High-performance caching** ([`GPCache`](@ref)) with in-place matrix operations to minimise allocations during inference.
- **Hyperparameter optimisation** utilities ([`neg_log_backwardmess_fast`](@ref), [`grad_llh_default!`](@ref)) for gradient-based tuning of kernel parameters.

## How to get started?

Head to the [Getting started](@ref getting-started) section.

## Table of Contents

```@contents
Pages = [
    "manuals/getting-started.md",
    "library/nodes.md",
    "library/meta.md",
    "library/kernels.md",
    "library/approximation.md",
    "library/prediction.md",
    "library/cache.md",
    "library/helpers.md",
    "library/types.md",
    "examples/overview.md",
    "contributing/guide.md",
]
Depth = 2
```

## Index

```@index
```
