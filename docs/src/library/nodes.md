```@meta
CurrentModule = RxGP
```

# [Factor nodes](@id nodes-reference)

RxGP defines three sparse Gaussian process factor nodes. Each node represents a local GP likelihood factor in a factor graph, with message passing rules defined on every edge.

## [Univariate sparse GP node](@id nodes-unisgp)

The [`UniSGP`](@ref) node implements a variational sparse Gaussian process (VSGP) factor for scalar observations. It is the special case of the dID-VSGP node (see [`UniSGP_dID`](@ref)) with an **identity operator** ``\mathcal{L} = \mathcal{I}``, so that the observation is a direct (noisy) evaluation of the latent GP.

The composite node function is obtained by collapsing the GP prior into the observation model (cf. Eq. 22 of the dID-VSGP derivation with ``\mathcal{L} = \mathcal{I}``):

```math
\tilde{\phi}(y, x, \mathbf{v}, w, \boldsymbol{\theta})
= \exp\!\Bigl(-\tfrac{1}{2}\, w\, A_u(x, \boldsymbol{\theta})\Bigr)\;
  \mathcal{N}\!\bigl(y \mid b_u(x, \mathbf{v}, \boldsymbol{\theta}),\; w^{-1}\bigr) ,
```

with the prior mean and residual variance

```math
b_u(x, \mathbf{v}, \boldsymbol{\theta})
  = m(x) + \mathbf{k}_{xu}(x, \boldsymbol{\theta})^\top
    \bigl(\mathbf{v} - K_{uu}(\boldsymbol{\theta})^{-1}\mathbf{m}_u\bigr) ,
```
```math
A_u(x, \boldsymbol{\theta})
  = k_{\boldsymbol{\theta}}(x, x)
    - \mathbf{k}_{xu}(x, \boldsymbol{\theta})^\top
      K_{uu}(\boldsymbol{\theta})^{-1}
      \mathbf{k}_{xu}(x, \boldsymbol{\theta}) ,
```

where ``\mathbf{v} = K_{uu}^{-1}\mathbf{u}`` is the transformed inducing variable, ``\mathbf{k}_{xu}(x) = k_{\boldsymbol{\theta}}(X_u, x) \in \mathbb{R}^M`` is the cross-kernel vector, ``K_{uu} = k_{\boldsymbol{\theta}}(X_u, X_u)``, ``\mathbf{m}_u = m(X_u)``, ``w`` is the noise precision, and ``m(x)`` is the mean function.

```@docs
UniSGP
```

**Edges:**

| Edge | Symbol | Type | Description |
|:-----|:-------|:-----|:------------|
| Output | `out` | `ℝ` | Observed (or latent) function value |
| Input | `in` | `ℝᴰ` | Input location (point or distribution) |
| Inducing variable | `v` | `ℝᴹ` | Transformed inducing variable ``K_{uu}^{-1}\mathbf{u}`` |
| Noise precision | `w` | `ℝ⁺` | Precision of the observation noise |
| Hyperparameters | `θ` | `ℝᵈ` | Kernel hyperparameters |

### Message passing rules — UniSGP

| Edge | Outgoing message | Summary |
|:-----|:-----------------|:--------|
| `out` | `NormalMeanPrecision` | Predictive mean from inducing variables projected to the observation |
| `in` | `ContinuousUnivariateLogPdf` | Backward message for input inference (log-pdf form) |
| `v` | `BufferUniSGP{MvNormalWeightedMeanPrecision}` | Inducing variable update with Cholesky tracking |
| `w` | `GammaShapeRate` | Gamma update for the noise precision |
| `θ` | `ContinuousUnivariateLogPdf` or `ContinuousMultivariateLogPdf` | Log-pdf for hyperparameter optimisation |

---

## [Univariate sparse GP node with inter-domain observations](@id nodes-unisgp-did)

The [`UniSGP_dID`](@ref) node implements a **decoupled inter-domain variational sparse Gaussian process** (dID-VSGP) factor. It generalises [`UniSGP`](@ref) by applying an arbitrary deterministic linear operator ``\mathcal{L}`` to the latent GP, so that observations live in the *transformed* space ``\tilde{f}(x) = \mathcal{L} f(x) \in \mathbb{R}^P`` (e.g. gradients, joint function-and-gradient stacks, or differential-equation constraints).

The composite node function (Eq. 22 in the paper) collapses the transformed GP prior into the observation model:

```math
\tilde{\phi}(\tilde{\mathbf{y}}, x, \mathbf{v}, W, \boldsymbol{\theta})
= \exp\!\Bigl(-\tfrac{1}{2}\operatorname{tr}\bigl(W\,\tilde{A}_u(x, \boldsymbol{\theta})\bigr)\Bigr)\;
  \mathcal{N}\!\bigl(\tilde{\mathbf{y}} \mid \tilde{b}_u(x, \mathbf{v}, \boldsymbol{\theta}),\; W^{-1}\bigr) ,
```

with the transformed prior mean and residual covariance

```math
\tilde{b}_u(x, \mathbf{v}, \boldsymbol{\theta})
  = \tilde{m}(x)
    + \tilde{K}_{xu}(x, \boldsymbol{\theta})\,
      \bigl(\mathbf{v} - K_{uu}(\boldsymbol{\theta})^{-1}\mathbf{m}_u\bigr)
  \in \mathbb{R}^P ,
```
```math
\tilde{A}_u(x, \boldsymbol{\theta})
  = \tilde{K}_{xx'}(x, x, \boldsymbol{\theta})
    - \tilde{K}_{xu}(x, \boldsymbol{\theta})\,
      K_{uu}(\boldsymbol{\theta})^{-1}\,
      \tilde{K}_{ux}(x, \boldsymbol{\theta})
  \in \mathbb{R}^{P \times P} ,
```

where the transformed kernels are

| Symbol | Definition | Size |
|:-------|:-----------|:-----|
| ``\tilde{m}(x)`` | ``\mathcal{L}\,m(x)`` | ``\mathbb{R}^P`` |
| ``\tilde{K}_{xu}(x, \boldsymbol{\theta})`` | ``\mathcal{L}_1\,k_{\boldsymbol{\theta}}(x, X_u)`` | ``\mathbb{R}^{P \times M}`` |
| ``\tilde{K}_{ux}(x, \boldsymbol{\theta})`` | ``\mathcal{L}_2\,k_{\boldsymbol{\theta}}(X_u, x)`` | ``\mathbb{R}^{M \times P}`` |
| ``\tilde{K}_{xx'}(x, x', \boldsymbol{\theta})`` | ``\mathcal{L}_1 \mathcal{L}_2\,k_{\boldsymbol{\theta}}(x, x')`` | ``\mathbb{R}^{P \times P}`` |
| ``K_{uu}(\boldsymbol{\theta})`` | ``k_{\boldsymbol{\theta}}(X_u, X_u)`` | ``\mathbb{R}^{M \times M}`` |

The inducing variables ``\mathbf{v} = K_{uu}^{-1}\mathbf{u}`` remain in the **latent** (untransformed) function space. This *decoupled* design allows multiple dID-VSGP nodes with different linear operators to share a single set of inducing variables in the same graph.

The operator is configured via the `operator` keyword in [`get_UniSGPMeta`](@ref) (`:grad` for ``P=D`` gradient observations, `:joint_fn_grad` for ``P=1+D`` stacked function-value-and-gradient observations).

```@docs
UniSGP_dID
```

**Edges:**

| Edge | Symbol | Type | Description |
|:-----|:-------|:-----|:------------|
| Output | `out` | `ℝᴾ` | Observed gradient (or stacked value + gradient) |
| Input | `in` | `ℝᴰ` | Input location |
| Inducing variable | `v` | `ℝᴹ` | Transformed inducing variable |
| Noise precision | `Wg` | `ℝᴾˣᴾ` | Wishart-distributed precision matrix for gradient noise |
| Hyperparameters | `θ` | `ℝᵈ` | Kernel hyperparameters |

### Message passing rules — UniSGP_dID

| Edge | Outgoing message | Summary |
|:-----|:-----------------|:--------|
| `out` | `NormalMeanPrecision` or `MvNormalMeanPrecision` | Predictive mean in the operator-transformed output space |
| `in` | `ContinuousUnivariateLogPdf` or `ContinuousMultivariateLogPdf` | Backward message for input |
| `v` | `BufferUniSGP{MvNormalWeightedMeanPrecision}` | Inducing variable update |
| `Wg` | `WishartFast` | Wishart update for the gradient noise precision |
| `θ` | `ContinuousUnivariateLogPdf` or `ContinuousMultivariateLogPdf` | Log-pdf for hyperparameters |

---

## [Multivariate sparse GP node](@id nodes-multisgp)

The [`MultiSGP`](@ref) node implements a multivariate variational sparse Gaussian process (VSGP) factor for **vector-valued** observations ``\mathbf{y} \in \mathbb{R}^D``. The multi-output structure is constructed via the *intrinsic coregionalization model* (ICM) with the coregionalization matrix ``C_c = I_D``, giving the matrix-valued kernel ``\mathcal{K}(\mathbf{x}, \mathbf{x}') = C_c \otimes k_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{x}')``.

The composite node function collapses the GP prior into the observation model:

```math
f_{\mathrm{SGP}}(\mathbf{y}, \mathbf{x}, \mathbf{v}, W, \boldsymbol{\theta})
= \exp\!\Bigl(-\tfrac{1}{2}\operatorname{tr}(W\, A_{\mathbf{x},\boldsymbol{\theta}})\Bigr)\;
  \mathcal{N}\!\bigl(\mathbf{y} \mid B_{\mathbf{x},\boldsymbol{\theta}}\,\mathbf{v},\; W^{-1}\bigr)
```

with the Kronecker-structured projection and residual covariance

```math
B_{\mathbf{x},\boldsymbol{\theta}} = C_c \otimes k_{\boldsymbol{\theta}}(\mathbf{x}, X_u)
  \in \mathbb{R}^{D \times DM},
```
```math
A_{\mathbf{x},\boldsymbol{\theta}}
  = C_c \otimes k_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{x})
    - B_{\mathbf{x},\boldsymbol{\theta}}\,
      K_u^{-1}\,
      B_{\mathbf{x},\boldsymbol{\theta}}^\top
  \in \mathbb{R}^{D \times D},
```

where ``K_u = C_c \otimes k_{\boldsymbol{\theta}}(X_u, X_u) \in \mathbb{R}^{DM \times DM}`` and ``\mathbf{v} = K_u^{-1}\mathbf{u} \in \mathbb{R}^{DM}``. Computational complexity is ``\mathcal{O}(DNM^2)``.

```@docs
MultiSGP
```

**Edges:**

| Edge | Symbol | Type | Description |
|:-----|:-------|:-----|:------------|
| Output | `out` | `ℝᴰ` | Observed vector output |
| Input | `in` | `ℝᴰˣ` | Input location |
| Inducing variable | `v` | `ℝᴰᴹ` | Stacked transformed inducing variables ``K_u^{-1}\mathbf{u}`` |
| Noise precision | `w` | `ℝᴰˣᴰ` | Wishart-distributed noise precision matrix |
| Hyperparameters | `θ` | `ℝᵈ` | Kernel hyperparameters |

### Message passing rules — MultiSGP

| Edge | Outgoing message | Summary |
|:-----|:-----------------|:--------|
| `out` | `MvNormalMeanPrecision` | Predictive mean via Kronecker-structured projection |
| `in` | `ContinuousUnivariateLogPdf` or `ContinuousMultivariateLogPdf` | Backward message for input |
| `v` | `MvNormalWeightedMeanPrecision` | Inducing variable update with Kronecker precision |
| `w` | `Wishart` | Wishart update for the noise precision |
| `θ` | `ContinuousMultivariateLogPdf` | Log-pdf for hyperparameters |

## [Average energy](@id nodes-average-energy)

Each node also defines an `@average_energy` method, used to evaluate the variational free energy (ELBO) contribution of the GP factor during inference. These are computed automatically by the ReactiveMP engine — no user action is required.

## [Node type traits](@id nodes-traits)

The `@node` macro automatically registers each node with ReactiveMP's type system:

```@docs
ReactiveMP.is_predefined_node(::Type{<:UniSGP})
ReactiveMP.is_predefined_node(::Type{<:UniSGP_dID})
ReactiveMP.is_predefined_node(::Type{<:MultiSGP})
```
