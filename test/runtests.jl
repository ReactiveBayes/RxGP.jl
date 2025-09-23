module TestRxGP
    include("helper_functions/approximate_kernel.jl")
    include("helper_functions/derivative.jl")
    include("helper_functions/gp_cache.jl")
    include("helper_functions/other_functions.jl")

    include("node_rule/multivariate.jl")
    include("node_rule/univariate.jl")
    
end
