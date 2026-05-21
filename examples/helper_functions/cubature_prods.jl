# This file is intended to be included standalone when working with non-conjugate products in RxInfer.jl.
# We include it here to showcase our examples that need to combine non-conjugate messages from the univariate
# and univariate_dID nodes with Guassian priors.

function ReactiveMP.prod(::GenericProd, left::NormalDistributionsFamily, right::Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}) 
    if left isa MultivariateGaussianDistributionsFamily
        m,c = ReactiveMP.approximate_meancov(srcubature(), (x) -> exp(right.logpdf(x)), left)
        if isnan(m[1])
            return left 
        else
            return MvNormalMeanCovariance(m,c + 1e-6I)
        end
    elseif left isa UnivariateGaussianDistributionsFamily
        m,v = ReactiveMP.approximate_meancov(ghcubature(21), (x) -> exp(right.logpdf(x)), left)
        if isnan(m) || isnan(v)
            return left 
        else
            return NormalMeanVariance(m,v + 1e-6)
        end
    end
end
function ReactiveMP.prod(::GenericProd, left::Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}, right::NormalDistributionsFamily) 
    if right isa MultivariateGaussianDistributionsFamily
        m,c = ReactiveMP.approximate_meancov(srcubature(), (x) -> exp(left.logpdf(x)), right)
        if isnan(m[1])
            return right
        else
            return MvNormalMeanCovariance(m,c + 1e-6I)
        end
    elseif right isa UnivariateGaussianDistributionsFamily
        m,v = ReactiveMP.approximate_meancov(ghcubature(21), (x) -> exp(left.logpdf(x)), right)
        if isnan(m) || isnan(v)
            return right 
        else
            return NormalMeanVariance(m,v + 1e-6)
        end
    end
end
function ReactiveMP.prod(::GenericProd, prod_of::ProductOf, right::NormalDistributionsFamily) 
    if right isa MultivariateGaussianDistributionsFamily
        m,c = ReactiveMP.approximate_meancov(srcubature(), (x) -> exp(prod_of.left.logpdf(x) + prod_of.right.logpdf(x)), right)
        if isnan(m[1])
            return right
        else
            return MvNormalMeanCovariance(m,c + 1e-6I)
        end
    elseif right isa UnivariateGaussianDistributionsFamily
        m,v = ReactiveMP.approximate_meancov(ghcubature(21), (x) -> exp(prod_of.left.logpdf(x) + prod_of.right.logpdf(x)), right)
        if isnan(m) || isnan(v)
            return right 
        else
            return NormalMeanVariance(m,v + 1e-6)
        end
    end
end
function ReactiveMP.prod(::GenericProd, left::NormalDistributionsFamily, prod_of::ProductOf) 
    if left isa MultivariateGaussianDistributionsFamily
        m,c = ReactiveMP.approximate_meancov(srcubature(), (x) -> exp(prod_of.left.logpdf(x) + prod_of.right.logpdf(x)), left)
        if isnan(m[1])
            return left 
        else
            return MvNormalMeanCovariance(m,c + 1e-6I)
        end
    elseif left isa UnivariateGaussianDistributionsFamily
        m,v = ReactiveMP.approximate_meancov(ghcubature(21), (x) -> exp(prod_of.left.logpdf(x) + prod_of.right.logpdf(x)), left)
        if isnan(m) || isnan(v)
            return left 
        else
            return NormalMeanVariance(m,v + 1e-6)
        end
    end
end
function ReactiveMP.prod(::GenericProd, left::Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}, right::Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}) 
    return ProductOf(left, right)
end

default_prod_rule(::NormalDistributionsFamily, ::Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}) = GenericProd()
default_prod_rule(::Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}, ::NormalDistributionsFamily) = GenericProd()
default_prod_rule(::Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}, ::Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}) = GenericProd()
default_prod_rule(::NormalDistributionsFamily, ::ProductOf) = GenericProd()
default_prod_rule(::ProductOf, ::NormalDistributionsFamily) = GenericProd()