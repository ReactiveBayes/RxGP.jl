# [Contributing](@id contributing-guide)

Contributions to RxGP.jl are welcome! This page covers how to get started.

## Setting up a development environment

1. Clone the repository:
   ```bash
   git clone https://github.com/ReactiveBayes/RxGP.jl.git
   cd RxGP.jl
   ```

2. Activate and instantiate the project:
   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.instantiate()
   ```

3. Run the test suite:
   ```julia
   Pkg.test()
   ```

## Project structure

| Directory | Contents |
|:----------|:---------|
| `src/` | Main source code |
| `src/SparseGaussianProcessnode/` | Node definitions and average energy computations |
| `src/rule/` | Message passing update rules, organised by node and edge |
| `src/helper_functions/` | Kernels, caching, approximation, prediction, meta, utilities |
| `test/` | Unit tests mirroring the `src/` structure |
| `examples/` | Jupyter notebook examples |
| `docs/` | Documentation (Documenter.jl) |

## Adding a new message passing rule

1. Create or edit the appropriate file in `src/rule/<node_type>_rules/<edge>.jl`.
2. Use the `@rule` macro from ReactiveMP.jl to define the rule.
3. Add corresponding tests in `test/node_rule/`.
4. Make sure `Pkg.test()` passes.

## Adding a new node

1. Define the node struct and register it with `@node` in `src/SparseGaussianProcessnode/`.
2. Define an `@average_energy` method.
3. Implement rules for each edge.
4. Create a corresponding meta type if needed.
5. Export the node and meta from the main module.
6. Add tests and update documentation.

## Building the documentation locally

```julia
using Pkg
Pkg.activate("docs")
Pkg.instantiate()
include("docs/make.jl")
```

The built documentation will be in `docs/build/`.
