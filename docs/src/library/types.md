# [Type aliases](@id types-reference)

RxGP defines several type aliases used throughout the message passing rules to constrain method signatures.

| Alias | Union members | Used for |
|:------|:-------------|:---------|
| `IN_OUT` | `Real`, `Array{<:Number}`, `PointMass`, `UnivariateNormalDistributionsFamily`, `MultivariateNormalDistributionsFamily` | Input and output edge types |
| `NOISE_Wg` | `Wishart`, `PointMass` | Gradient noise precision |
| `NOISE_w` | `GammaShapeRate`, `PointMass` | Scalar noise precision |
| `TYPE_Norm` | `UnivariateNormalDistributionsFamily`, `MultivariateNormalDistributionsFamily` | Any normal distribution |
| `TYPE_ContLogPdf` | `ContinuousUnivariateLogPdf`, `ContinuousMultivariateLogPdf` | Log-pdf message types |

These aliases are not exported but are used internally to dispatch the correct message passing rules.
