using TestItemRunner

# Include test files to register @testitem blocks
include("helper_functions/approximate_kernel.jl")
include("helper_functions/derivative.jl")
include("helper_functions/gp_cache.jl")
include("helper_functions/other_functions.jl")
include("helper_functions/univariate_prediction_fns.jl")

include("node_rule/multivariate.jl")
include("node_rule/univariate.jl")
include("node_rule/univariate_grad.jl")

# Run all registered test items when using Pkg.test()
@run_package_tests
