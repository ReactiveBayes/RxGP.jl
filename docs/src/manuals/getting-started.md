```@meta
CurrentModule = RxGP
```

# [Getting started](@id getting-started)

This guide walks you through installing RxGP, building a simple sparse GP regression model, and running inference.

## Installation

```julia
using Pkg
Pkg.add("RxGP")
```

RxGP depends on [RxInfer.jl](https://github.com/ReactiveBayes/RxInfer.jl), [ReactiveMP.jl](https://github.com/ReactiveBayes/ReactiveMP.jl), and [KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl), which will be installed automatically.

## Overview of the approach

RxGP implements **Sparse Variational Gaussian Process** (SVGP) inference as factor nodes in a factor graph. The key idea is:

1. Place a small set of **inducing points** ``\mathbf{X}_u = \{x_{u,1}, \ldots, x_{u,M}\}`` in the input space.
2. Define a **transformed inducing variable** ``\mathbf{v} = K_{uu}^{-1}\mathbf{u}``, where ``\mathbf{u} = f(\mathbf{X}_u)`` and ``K_{uu} = k(\mathbf{X}_u, \mathbf{X}_u)``.
3. For each observation ``y_n``, create a [`UniSGP`](@ref) node that connects the output ``y_n``, the input ``x_n``, the shared inducing variable ``\mathbf{v}``, the noise precision ``w``, and the kernel hyperparameters ``\theta``.
4. Run variational message passing to infer posterior beliefs over all latent variables.

## Quick example: GP regression

```julia
using RxInfer, RxGP, KernelFunctions, Plots, Random

# --- Generate data ---
Random.seed!(42)
N = 50
x_train = [rand(1) * 10 for _ in 1:N]           # 1D inputs
f_true(x) = sin(x[1])
y_train = [f_true(x) + 0.1 * randn() for x in x_train]

# --- Set up the GP ---
D = 1                                              # input dimension
M = 15                                             # number of inducing points
Xu = [collect(range(0, 10, length=M)[i:i]) for i in 1:M]  # inducing locations

# Kernel: squared-exponential with initial hyperparameters
kernel_fn, θ_init, dim_θ = get_simple_kernel_and_params(D; kernel_spec=:SE)
θ_fixed = θ_init

# Build meta object
meta = get_UniSGPMeta(D;
    method    = GenUnscented(),
    mean_fn   = (x) -> 0.0,
    kernel    = kernel_fn,
    kernel_spec = :SE,
    operator  = :fn,
    Xu        = Xu,
    θ         = θ_fixed,
)

# --- Define the model ---
@model function gp_regression(y, x, meta)
    θ  ~ PointMass(θ_fixed)
    w  ~ GammaShapeRate(1.0, 1.0)
    v  ~ MvNormalMeanCovariance(zeros(M), diagm(ones(M)))
    for n in eachindex(y)
        y[n] ~ UniSGP(x[n], v, w, θ) where { meta = meta }
    end
end

# --- Run inference ---
result = infer(
    model = gp_regression(meta = meta),
    data  = (y = y_train, x = x_train),
    iterations = 10,
)

# --- Predict ---
x_test = [[xi] for xi in range(0, 10, length=100)]
q_v = result.posteriors[:v][end]
means, covs = predict_GP(m_in=x_test, q_v=q_v, q_θ=PointMass(θ_fixed), meta=meta)
```

!!! note
    The example above uses fixed kernel hyperparameters. For hyperparameter optimisation, see the gradient-based utilities [`neg_log_backwardmess_fast`](@ref) and [`grad_llh_default!`](@ref), which can be combined with [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl).

## What's next?

- Learn about the three GP factor nodes: [`UniSGP`](@ref nodes-reference), [`UniSGP_dID`](@ref nodes-reference), and [`MultiSGP`](@ref nodes-reference).
- See how to configure kernel functions and meta objects in [Kernel specification](@ref kernels-reference) and [Meta objects](@ref meta-reference).
- Explore the [Examples](@ref examples-overview) for GP regression, GP state-space models, and gradient observations.
