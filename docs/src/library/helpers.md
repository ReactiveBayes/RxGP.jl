```@meta
CurrentModule = RxGP
```

# [Helper utilities](@id helpers-reference)

RxGP exports several utility functions used throughout the message passing rules and by end users.

## Mean function application

```@docs
apply_mean_fn
```

Applies the scalar mean function `mf` to an input `x`. Handles dispatch for `PointMass`, `Distribution`, scalar, and vector inputs.

## Mean and covariance extraction

```@docs
mean_cov_scalar_matrix
mean_cov_vector_matrix
```

These convenience functions extract `(mean, cov)` from various input types (`PointMass`, `Real`, distribution) and normalise the output shapes:

| Function | Returns |
|:---------|:--------|
| `mean_cov_scalar_matrix` | `(scalar, 1×1 matrix)` |
| `mean_cov_vector_matrix` | `(vector, matrix)` |

## Linear algebra helpers

```@docs
jdotavx
create_blockmatrix
```

## Inducing variable buffer

```@docs
BufferUniSGP
```

`BufferUniSGP` wraps an inducing-variable message together with its meta object. When the product of all incoming messages to `v` has been accumulated (tracked via `meta.counter`), it updates the Cholesky factor of the second moment `meta.Uv`. This ensures that the free-energy computation remains efficient.

## Hyperparameter optimisation

The following functions compute the negative log backward message and its gradient with respect to the kernel hyperparameters ``\theta``. They are designed to be used with gradient-based optimisers (e.g. [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) or [Flux.jl](https://github.com/FluxML/Flux.jl)).

```@docs
neg_log_backwardmess_fast
neg_log_backwardmess_uncertain
neg_log_backwardmess_msg
grad_llh_default!
neg_log_backwardmess_multi
grad_llh_multi!
```
