module RxGP

using RxInfer, LoopVectorization, ReactiveMP
using Zygote, Optim, ForwardDiff, StatsFuns, KernelFunctions, LinearAlgebra
using Distributions, Random, SparseArrays

import ReactiveMP: getweights, getpoints, approximate_meancov, WishartFast

# other
include("types.jl")

# helper functions
include("helper_functions/genUT.jl")
include("helper_functions/approximate_kernel.jl")
include("helper_functions/derivative.jl")
include("helper_functions/gp_cache.jl")
include("helper_functions/other_functions.jl")
include("helper_functions/meta.jl")
include("helper_functions/common.jl")
include("helper_functions/univariate_prediction_fns.jl")

# node 
include("SparseGaussianProcessnode/univariate_node.jl")
include("SparseGaussianProcessnode/univariate_dID_node.jl")
include("SparseGaussianProcessnode/multivariate_node.jl")

# rule
include("rule/univariate_rules/out.jl")
include("rule/univariate_rules/in.jl")
include("rule/univariate_rules/v.jl")
include("rule/univariate_rules/w.jl")
include("rule/univariate_rules/theta.jl")

include("rule/univariate_dID_rules/out.jl")
include("rule/univariate_dID_rules/in.jl")
include("rule/univariate_dID_rules/v.jl")
include("rule/univariate_dID_rules/Wg.jl")
include("rule/univariate_dID_rules/theta.jl")

include("rule/multivariate_rules/out.jl")
include("rule/multivariate_rules/in.jl")
include("rule/multivariate_rules/v.jl")
include("rule/multivariate_rules/w.jl")
include("rule/multivariate_rules/theta.jl")

end