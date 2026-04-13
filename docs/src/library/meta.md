```@meta
CurrentModule = RxGP
```

# [Meta objects](@id meta-reference)

Meta objects carry all the precomputed and configuration data that the GP factor nodes need during message passing. They are attached to nodes using the `where { meta = ... }` syntax in model specification.

## UniSGPMeta

[`UniSGPMeta`](@ref) is used with both [`UniSGP`](@ref) and [`UniSGP_dID`](@ref) nodes. It stores:

- The approximation method (e.g. [`GenUnscented`](@ref))
- The mean function, kernel, and inducing point locations
- Precomputed kernel matrices and Cholesky factors
- The operator-specific functions (`Lm_fn`, `Kxu_fn`, `Kxx_fn`) for identity, gradient, or joint operators
- Mutable workspace for kernel expectations (``\Psi`` statistics) updated during inference
- A running Cholesky factor of the second moment of ``\mathbf{v}``

```@docs
UniSGPMeta
```

### Construction

The recommended way to create a [`UniSGPMeta`](@ref) is via [`get_UniSGPMeta`](@ref):

```@docs
get_UniSGPMeta
```

**Key arguments:**

| Argument | Default | Description |
|:---------|:--------|:------------|
| `D` | (positional) | Input dimensionality |
| `method` | `nothing` | Approximation method (e.g. `GenUnscented()`) for uncertain inputs |
| `mean_fn` | (required) | Prior mean function `m(x) → ℝ` |
| `kernel` | (required) | Kernel constructor `θ → Kernel` (from KernelFunctions.jl) |
| `kernel_spec` | `:SE` | Kernel family: `:SE`, `:SEn`, `:SMn`, or `:SEn_SMn` |
| `mode` | `:AD` | Differentiation mode: `:AD` (autodiff) or `:AN` (analytic, SE only) |
| `operator` | `:fn` | Linear operator: `:fn` (identity, P=1), `:grad` (gradient, P=D), `:joint_fn_grad` (P=1+D) |
| `Xu` | (required) | Vector of inducing point locations |
| `θ` | (required) | Initial kernel hyperparameters |
| `Lm_fn`, `Kxu_fn`, `Kxx_fn` | `nothing` | Override operator functions for a custom linear operator |

### Accessors

```@docs
getmethod
getMeanFn
getInducingInput
getΨx
getΨxx
getΨ0
getΨ1_trans
getΨ2
getΨ3
getLm_fn
getKxx_fn
getKxu_fn
getKuuF
getKernel
get_dims_data
get_dims_theta
getUv
getcounter
getN
```

---

## MultiSGPMeta

[`MultiSGPMeta`](@ref) is used with the [`MultiSGP`](@ref) node.

```@docs
MultiSGPMeta
```

### Accessors

```@docs
getKuuInverse
getGPCache
```

!!! note
    Several accessors (`getInducingInput`, `getΨ0`, `getΨ1_trans`, `getΨ2`, `getKernel`, `getmethod`) are shared between `UniSGPMeta` and `MultiSGPMeta`.
