# ================== Prod rule definitions ================== #
TYPE_Norm = Union{UnivariateNormalDistributionsFamily, MultivariateNormalDistributionsFamily}
TYPE_ContLogPdf = Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}

function ReactiveMP.prod(::GenericProd, left::TYPE_Norm, right::TYPE_ContLogPdf)
    return left 
    function total_logpdf(x) return logpdf(left, x) + logpdf(right, x) end
    function my_logpdf!(out, x)
        out[1] = total_logpdf(x)
    end
    function my_grad_hess!(out_grad, out_hess, x)
        out_grad .= ForwardDiff.gradient(total_logpdf, x)
        out_hess .= ForwardDiff.hessian(total_logpdf, x)
    end
    params = ProjectionParameters(
        tolerance = 1e-6,
        strategy = ExponentialFamilyProjection.GaussNewton(nsamples = 0),
    )
    inplace = ExponentialFamilyProjection.InplaceLogpdfGradHess(my_logpdf!, my_grad_hess!)
    prj = ProjectedTo(MvNormalMeanCovariance, length(left); parameters = params)
    projected_Gauss = project_to(prj, inplace)
    return projected_Gauss
end

function ReactiveMP.prod(::GenericProd, left::TYPE_ContLogPdf, right::TYPE_Norm)
    return right 
    function total_logpdf(x) return logpdf(left, x) + logpdf(right, x) end
    function my_logpdf!(out, x)
        out[1] = total_logpdf(x)
    end
    function my_grad_hess!(out_grad, out_hess, x)
        out_grad .= ForwardDiff.gradient(total_logpdf, x)
        out_hess .= ForwardDiff.hessian(total_logpdf, x)
    end
    params = ProjectionParameters(
        tolerance = 1e-6,
        strategy = ExponentialFamilyProjection.GaussNewton(nsamples = 0),
    )
    inplace = ExponentialFamilyProjection.InplaceLogpdfGradHess(my_logpdf!, my_grad_hess!)
    prj = ProjectedTo(MvNormalMeanCovariance, length(right); parameters = params)
    projected_Gauss = project_to(prj, inplace)
    return projected_Gauss
end

function ReactiveMP.prod(::GenericProd, left::TYPE_ContLogPdf, right::TYPE_ContLogPdf) 
    return ProductOf(left, right)
end

function ReactiveMP.prod(::GenericProd, left::TYPE_Norm, prod_of::ProductOf)
    return left 
    function total_logpdf(x) return logpdf(left, x) + logpdf(prod_of.left, x) + logpdf(prod_of.right, x) end
    function my_logpdf!(out, x)
        out[1] = total_logpdf(x)
    end
    function my_grad_hess!(out_grad, out_hess, x)
        out_grad .= ForwardDiff.gradient(total_logpdf, x)
        out_hess .= ForwardDiff.hessian(total_logpdf, x)
    end
    params = ProjectionParameters(
        tolerance = 1e-6,
        strategy = ExponentialFamilyProjection.GaussNewton(nsamples = 0),
    )
    inplace = ExponentialFamilyProjection.InplaceLogpdfGradHess(my_logpdf!, my_grad_hess!)
    prj = ProjectedTo(MvNormalMeanCovariance, length(left); parameters = params)
    projected_Gauss = project_to(prj, inplace)
    return projected_Gauss
end

function ReactiveMP.prod(::GenericProd, prod_of::ProductOf, right::TYPE_Norm) 
    return right
    function total_logpdf(x) return logpdf(prod_of.left, x) + logpdf(prod_of.right, x) + logpdf(right, x) end
    function my_logpdf!(out, x)
        out[1] = total_logpdf(x)
    end
    function my_grad_hess!(out_grad, out_hess, x)
        out_grad .= ForwardDiff.gradient(total_logpdf, x)
        out_hess .= ForwardDiff.hessian(total_logpdf, x)
    end
    params = ProjectionParameters(
        tolerance = 1e-6,
        strategy = ExponentialFamilyProjection.GaussNewton(nsamples = 0),
    )
    inplace = ExponentialFamilyProjection.InplaceLogpdfGradHess(my_logpdf!, my_grad_hess!)
    prj = ProjectedTo(MvNormalMeanCovariance, length(right); parameters = params)
    projected_Gauss = project_to(prj, inplace)
    return projected_Gauss
end

default_prod_rule(::TYPE_Norm, ::TYPE_ContLogPdf) = GenericProd()
default_prod_rule(::TYPE_ContLogPdf, ::TYPE_Norm) = GenericProd()
default_prod_rule(::TYPE_ContLogPdf, ::TYPE_ContLogPdf) = GenericProd()
default_prod_rule(::TYPE_Norm, ::ProductOf) = GenericProd()
default_prod_rule(::ProductOf, ::TYPE_Norm) = GenericProd()