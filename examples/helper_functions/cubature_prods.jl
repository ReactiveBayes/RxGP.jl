# This file is intended to be included standalone when working with non-conjugate products in RxInfer.jl.
# We include it here to showcase our examples that need to combine non-conjugate messages from the univariate
# and univariate_grad nodes with Guassian priors.

# multivariate case
function ReactiveMP.prod(::GenericProd, left::MultivariateGaussianDistributionsFamily, right::ContinuousMultivariateLogPdf) 
    m,c = approximate_meancov(srcubature(),(x) -> exp(right.logpdf(x)),left)
    if isnan(m[1])
        return left 
    else
        return MvNormalMeanCovariance(m,c + 1e-6I)
    end
end
function ReactiveMP.prod(::GenericProd, left::ContinuousMultivariateLogPdf, right::MultivariateGaussianDistributionsFamily) 
    m,c = approximate_meancov(srcubature(),(x) -> exp(left.logpdf(x)),right)
    if isnan(m[1])
        return right 
    else
        return MvNormalMeanCovariance(m,c + 1e-6I)
    end
end
function ReactiveMP.prod(::GenericProd, prod_of::ProductOf, right::MultivariateGaussianDistributionsFamily) 
    m,c = approximate_meancov(srcubature(),(x) -> exp(prod_of.left.logpdf(x) + prod_of.right.logpdf(x)),right)
    if isnan(m[1])
        return right
    else
        return MvNormalMeanCovariance(m,c + 1e-6I)
    end
end
function ReactiveMP.prod(::GenericProd, left::MultivariateGaussianDistributionsFamily, prod_of::ProductOf) 
    m,c = approximate_meancov(srcubature(),(x) -> exp(prod_of.left.logpdf(x) + prod_of.right.logpdf(x)),left)
    if isnan(m[1])
        return left 
    else
        return MvNormalMeanCovariance(m,c + 1e-6I)
    end
end
function ReactiveMP.prod(::GenericProd, left::ContinuousMultivariateLogPdf, right::ContinuousMultivariateLogPdf) 
    return ProductOf(left, right)
end

# univariate case 
function ReactiveMP.prod(::GenericProd, left::UnivariateGaussianDistributionsFamily, right::ContinuousUnivariateLogPdf) 
    m,v = approximate_meancov(ghcubature(21),(x) -> exp(right.logpdf(x)),left)
    if isnan(m) || isnan(v)
        return left 
    else
        return NormalMeanVariance(m,v + 1e-6)
    end
end
function ReactiveMP.prod(::GenericProd, left::ContinuousUnivariateLogPdf, right::UnivariateGaussianDistributionsFamily) 
    m,v = approximate_meancov(ghcubature(21),(x) -> exp(left.logpdf(x)),right)
    if isnan(m) || isnan(v)
        return right
    else
        return NormalMeanVariance(m,v + 1e-6)
    end
end
function ReactiveMP.prod(::GenericProd, prod_of::ProductOf, right::UnivariateGaussianDistributionsFamily) 
    m,v = approximate_meancov(ghcubature(21),(x) -> exp(prod_of.left.logpdf(x) + prod_of.right.logpdf(x)),right)
    if isnan(m) || isnan(v)
        return right
    else
        return NormalMeanVariance(m,v + 1e-6)
    end
end
function ReactiveMP.prod(::GenericProd, left::UnivariateGaussianDistributionsFamily, prod_of::ProductOf) 
    m,v = approximate_meancov(ghcubature(21),(x) -> exp(prod_of.left.logpdf(x) + prod_of.right.logpdf(x)),left)
    if isnan(m) || isnan(v)
        return left 
    else
        return NormalMeanVariance(m,v + 1e-6)
    end
end
function ReactiveMP.prod(::GenericProd, left::ContinuousUnivariateLogPdf, right::ContinuousUnivariateLogPdf) 
    return ProductOf(left, right)
end

# shared
default_prod_rule(::NormalDistributionsFamily, ::Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}) = GenericProd()
default_prod_rule(::Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}, ::NormalDistributionsFamily) = GenericProd()
default_prod_rule(::Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}, ::Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}) = GenericProd()
default_prod_rule(::NormalDistributionsFamily, ::ProductOf) = GenericProd()
default_prod_rule(::ProductOf, ::NormalDistributionsFamily) = GenericProd()