using ExponentialFamilyProjection

# This file is intended to be included standalone when working with non-conjugate products in RxInfer.jl.
# We include it here to showcase our examples that need to combine non-conjugate messages from the univariate
# and univariate_grad nodes with Guassian priors.

# ================== Prod rule definitions ================== #
# ==== Multivariate ==== #

function ReactiveMP.prod(::GenericProd, left::NormalDistributionsFamily, right::Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf})
    dims = length(left)
    function total_logpdf(x) return dims == 1 ? logpdf(left, only(x)) + logpdf(right, only(x)) : logpdf(left, x) + logpdf(right, x) end
    function my_logpdf!(out, x)
        out[1] = total_logpdf(x)
    end
    function my_grad_hess!(out_grad, out_hess, x)
        out_grad .= ForwardDiff.gradient(total_logpdf, x)
        out_hess .= ForwardDiff.hessian(total_logpdf, x)
    end
    params = ProjectionParameters(
        tolerance = 1e-6,
        niterations = 2000,
        strategy = ExponentialFamilyProjection.GaussNewton(nsamples = 0),
    )
    inplace = ExponentialFamilyProjection.InplaceLogpdfGradHess(my_logpdf!, my_grad_hess!)
    prj = ProjectedTo(MvNormalMeanCovariance, dims; parameters = params)
    projected_Gauss = project_to(prj, inplace)
    return dims == 1 ? NormalMeanVariance(only(mean(projected_Gauss)), only(cov(projected_Gauss))) : projected_Gauss
end

function ReactiveMP.prod(::GenericProd, left::Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}, right::NormalDistributionsFamily)
    dims = length(right)
    function total_logpdf(x) return dims == 1 ? logpdf(left, only(x)) + logpdf(right, only(x)) : logpdf(left, x) + logpdf(right, x) end
    function my_logpdf!(out, x)
        out[1] = total_logpdf(x)
    end
    function my_grad_hess!(out_grad, out_hess, x)
        out_grad .= ForwardDiff.gradient(total_logpdf, x)
        out_hess .= ForwardDiff.hessian(total_logpdf, x)
    end
    params = ProjectionParameters(
        tolerance = 1e-6,
        niterations = 2000,
        strategy = ExponentialFamilyProjection.GaussNewton(nsamples = 0),
    )
    inplace = ExponentialFamilyProjection.InplaceLogpdfGradHess(my_logpdf!, my_grad_hess!)
    prj = ProjectedTo(MvNormalMeanCovariance, dims; parameters = params)
    projected_Gauss = project_to(prj, inplace)
    return dims == 1 ? NormalMeanVariance(only(mean(projected_Gauss)), only(cov(projected_Gauss))) : projected_Gauss
end

function ReactiveMP.prod(::GenericProd, left::Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}, right::Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}) 
    return ProductOf(left, right)
end

function ReactiveMP.prod(::GenericProd, left::NormalDistributionsFamily, prod_of::ProductOf)
    dims = length(left)
    function total_logpdf(x) return dims == 1 ? logpdf(left, only(x)) + logpdf(prod_of.left, only(x)) + logpdf(prod_of.right, only(x)) : logpdf(left, x) + logpdf(prod_of.left, x) + logpdf(prod_of.right, x) end
    function my_logpdf!(out, x)
        out[1] = total_logpdf(x)
    end
    function my_grad_hess!(out_grad, out_hess, x)
        out_grad .= ForwardDiff.gradient(total_logpdf, x)
        out_hess .= ForwardDiff.hessian(total_logpdf, x)
    end
    params = ProjectionParameters(
        tolerance = 1e-6,
        niterations = 2000,
        strategy = ExponentialFamilyProjection.GaussNewton(nsamples = 0),
    )
    inplace = ExponentialFamilyProjection.InplaceLogpdfGradHess(my_logpdf!, my_grad_hess!)
    prj = ProjectedTo(MvNormalMeanCovariance, dims; parameters = params)
    projected_Gauss = project_to(prj, inplace)
    return dims == 1 ? NormalMeanVariance(only(mean(projected_Gauss)), only(cov(projected_Gauss))) : projected_Gauss
end

function ReactiveMP.prod(::GenericProd, prod_of::ProductOf, right::NormalDistributionsFamily) 
    dims = length(right)
    function total_logpdf(x) return dims == 1 ? logpdf(prod_of.left, only(x)) + logpdf(prod_of.right, only(x)) + logpdf(right, only(x)) : logpdf(prod_of.left, x) + logpdf(prod_of.right, x) + logpdf(right, x) end
    function my_logpdf!(out, x)
        out[1] = total_logpdf(x)
    end
    function my_grad_hess!(out_grad, out_hess, x)
        out_grad .= ForwardDiff.gradient(total_logpdf, x)
        out_hess .= ForwardDiff.hessian(total_logpdf, x)
    end
    params = ProjectionParameters(
        tolerance = 1e-6,
        niterations = 2000,
        strategy = ExponentialFamilyProjection.GaussNewton(nsamples = 0),
    )
    inplace = ExponentialFamilyProjection.InplaceLogpdfGradHess(my_logpdf!, my_grad_hess!)
    prj = ProjectedTo(MvNormalMeanCovariance, dims; parameters = params)
    projected_Gauss = project_to(prj, inplace)
    return dims == 1 ? NormalMeanVariance(only(mean(projected_Gauss)), only(cov(projected_Gauss))) : projected_Gauss
end

default_prod_rule(::NormalDistributionsFamily, ::Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}) = GenericProd()
default_prod_rule(::Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}, ::NormalDistributionsFamily) = GenericProd()
default_prod_rule(::Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}, ::Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}) = GenericProd()
default_prod_rule(::NormalDistributionsFamily, ::ProductOf) = GenericProd()
default_prod_rule(::ProductOf, ::NormalDistributionsFamily) = GenericProd()