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

Minimal GP regression with an EM loop for hyperparameter optimisation:

```julia
using RxInfer, RxGP, BayesBase
using KernelFunctions, LinearAlgebra, Flux

# --- Data ---
N  = 30
Nu = 20
Xu = [[x] for x in range(-4, 4; length=Nu)]      # inducing inputs
xtrain = sort(rand(N) .* 8 .- 4)                  # training inputs
ytrain = sinc.(xtrain) .+ 0.05 .* randn(N)        # noisy observations

# --- Kernel ---
D = 1
mean_fn = (x) -> 0.0
kernel, θ_init, _ = get_simple_kernel_and_params(D; kernel_spec=:SE)

# --- Model (priors passed as arguments for EM warm-starting) ---
@model function gp_regression(y, x, Xu, θ, mv_prior, gamma_prior)
    v ~ mv_prior
    w ~ gamma_prior
    for i in eachindex(y)
        y[i] ~ UniSGP(x[i], v, w, θ)
    end
end

@meta function gpr_meta(; Xu, θ)
    UniSGP() -> get_UniSGPMeta(D; mean_fn=mean_fn, kernel=kernel, operator=:fn, Xu=Xu, θ=θ)
end

@initialization function gpr_init(q_v_init, q_w_init)
    q(v) = q_v_init
    q(w) = q_w_init
end

gpr_constraints = @constraints begin
    q(v, w) = q(v)q(w)
end

# --- EM loop ---
θ_opt = deepcopy(θ_init)
q_v   = MvNormalWeightedMeanPrecision(zeros(Nu), 50 * diageye(Nu))
q_w   = GammaShapeRate(1e-2, 1e-2)
state = Flux.setup(Flux.AdaMax(0.01), θ_opt)
grad  = similar(θ_opt)

for run in 1:15
    # E-step: VMP inference, seeded from the previous posterior
    res = infer(
        model          = gp_regression(Xu=Xu, θ=θ_opt, mv_prior=q_v, gamma_prior=q_w),
        data           = (y=ytrain, x=[[x] for x in xtrain]),
        meta           = gpr_meta(Xu=Xu, θ=θ_opt),
        initialization = gpr_init(q_v, q_w),
        constraints    = gpr_constraints,
        returnvars     = KeepLast(),
        iterations     = 2,
        free_energy    = true,
    )
    q_v, q_w = res.posteriors[:v], res.posteriors[:w]

    # M-step: gradient update of kernel hyperparameters
    for _ in 1:10
        grad_llh_default!(grad, θ_opt;
            y_data=ytrain, x_y_data=[[x] for x in xtrain], x_ω_data=[[x] for x in xtrain],
            q_v=q_v, q_w=q_w, kernel=kernel, mean_fn=mean_fn, Xu=Xu)
        Flux.Optimise.update!(state, θ_opt, grad)
    end
end

# --- Predict ---
xtest = [[x] for x in range(-6, 6; length=200)]
meta_fn = get_UniSGPMeta(D; mean_fn=mean_fn, kernel=kernel, operator=:fn, Xu=Xu, θ=θ_opt)
means, covs = predict_GP(m_in=xtest, q_v=q_v, q_θ=θ_opt, meta=meta_fn)
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

## Examples

Jupyter notebooks are provided in the `examples/` directory. To run them, set up the dedicated environment once from the repository root:

```julia
using Pkg
Pkg.activate("examples")
Pkg.develop(PackageSpec(path="."))   # link local RxGP
Pkg.instantiate()
```

On subsequent sessions, only `Pkg.activate("examples")` is needed before opening a notebook.

## References

- H.M.H. Nguyen, İ. Şenöz, and B. de Vries, "A Factor Graph Approach to Variational Sparse Gaussian Processes," *IEEE Open Journal of Signal Processing*, 2025.
- A.H. Ledbetter, H.M.H. Nguyen, and B. de Vries, "A Factor-Graph Approach to Decoupled Inter-Domain Variational Sparse Gaussian Processes with Arbitrary Mean Functions," *Under review*, 2026.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file.
