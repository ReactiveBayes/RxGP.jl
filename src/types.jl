# GPMessageTypes for RxGP.jl

import ReactiveMP: PointMass, Wishart, GammaShapeRate,
                UnivariateNormalDistributionsFamily,
                MultivariateNormalDistributionsFamily,
                ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf

const IN_OUT = Union{Real, Array{<:Number}, PointMass, UnivariateNormalDistributionsFamily, MultivariateNormalDistributionsFamily}
const NOISE_Wg = Union{Wishart, PointMass}
const NOISE_w = Union{GammaShapeRate, PointMass}
const TYPE_Norm = Union{UnivariateNormalDistributionsFamily, MultivariateNormalDistributionsFamily}
const TYPE_ContLogPdf = Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}