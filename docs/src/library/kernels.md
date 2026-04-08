```@meta
CurrentModule = RxGP
```

# [Kernel specification](@id kernels-reference)

RxGP builds on [KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl) and provides utilities for constructing parameterised kernels, computing their gradient and Hessian operators, and initialising hyperparameters.

## Building a kernel

```@docs
get_simple_kernel_and_params
```

This function returns a tuple `(kernel_fn, θ_init, dim_θ)` where:

- `kernel_fn(θ)` returns a `KernelFunctions.Kernel` instance parameterised by `θ`.
- `θ_init` is a vector of initial hyperparameters (in softplus-transformed space).
- `dim_θ` is the total number of hyperparameters.

**Supported kernel families:**

| `kernel_spec` | Description | Parameters per component |
|:--------------|:------------|:------------------------|
| `:SE` | Single squared-exponential | `1 + D` (variance + lengthscales) or `2` (shared lengthscale) |
| `:SEn` | Sum of `num_SE` SE kernels | `num_SE × (1 + D)` or `num_SE × 2` |
| `:SMn` | Sum of `num_SM` spectral-mixture components | `num_SM × (1 + 2D)` |
| `:SEn_SMn` | Sum of SE and SM components | Combined |

The `independent_SE_lengthscales` flag controls whether each input dimension gets its own lengthscale (`true`, default) or all dimensions share one (`false`).

## Gradient and Hessian kernel functions

For derivative observation models (`:grad` and `:joint_fn_grad` operators), RxGP needs the gradient and Hessian of the kernel with respect to the input:

```@docs
get_gradient_Kxu_fn
get_gradient_Kxx_fn
```

These are called internally by [`get_UniSGPMeta`](@ref) when `operator` is `:grad` or `:joint_fn_grad`. You only need to call them directly for advanced custom setups.

**Modes:**

| Mode | Description | Supported kernels |
|:-----|:------------|:-----------------|
| `:AD` | Automatic differentiation via ForwardDiff.jl | All kernel families |
| `:AN` | Analytic closed-form expressions (~10× faster) | `:SE`, `:SEn` only |
