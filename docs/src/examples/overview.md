```@meta
CurrentModule = RxGP
```

# [Examples](@id examples-overview)

RxGP ships with Jupyter notebook examples in the `examples/` directory of the repository. These demonstrate end-to-end workflows for common GP modelling scenarios.

## GP Regression

**Notebook:** `examples/GPRegression.ipynb`

Standard sparse GP regression with scalar observations. Demonstrates:
- Setting up inducing points and a squared-exponential kernel
- Defining the model with [`UniSGP`](@ref) nodes
- Running variational inference with RxInfer
- Posterior prediction with [`predict_GP`](@ref)

## GP State-Space Model

**Notebook:** `examples/GPSSM.ipynb`

A Gaussian process state-space model where the GP governs the transition dynamics. Demonstrates:
- Composing GP factor nodes with standard state-space model components
- Streaming (online) inference over time steps
- Uncertain inputs flowing through the GP node

## GP Regression with Gradient Observations (AMF)

**Notebook:** `examples/GP_Regression_AMF_Grads.ipynb`

GP regression that incorporates **derivative observations** alongside function-value observations. Demonstrates:
- Using the [`UniSGP_dID`](@ref) node with `:grad` and `:joint_fn_grad` operators
- Automatic Mean Field (AMF) variational inference
- Hyperparameter optimisation via gradient-based methods

!!! tip
    To run the notebooks, clone the repository and open them with Jupyter or VS Code.
    The notebooks have their own dedicated environment in `examples/`. Set it up once from the
    repository root:
    ```julia
    using Pkg
    Pkg.activate("examples")
    Pkg.develop(PackageSpec(path="."))   # link local RxGP
    Pkg.instantiate()
    ```
    On subsequent sessions, only `Pkg.activate("examples")` is needed before opening a notebook.
