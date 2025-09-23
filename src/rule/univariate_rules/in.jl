# rule for "in" (univariate case) 
@rule UniSGP(:in, Marginalisation) (q_out::UnivariateNormalDistributionsFamily, q_v::MultivariateNormalDistributionsFamily,q_w::Any,q_θ::PointMass, meta::UniSGPMeta) = begin
    w_bar = mean(q_w)
    kernel = getKernel(meta)
    θ = mean(q_θ)
    B_trans = (x) -> kernelmatrix(kernel(θ), meta.Xu,[x])
    α = (x) -> meta.KuuL \ B_trans(x)
    A = (x) -> kernelmatrix(kernel(θ),[x]) .- dot(α(x),α(x))
    β = (x) -> meta.Uv * B_trans(x)
    
    μ_y, v_y = mean_var(q_out)
    μ_v = mean(q_v)

    log_backwardmess = (x) -> -0.5 * A(x)[1] * w_bar + w_bar * μ_y * dot(B_trans(x),μ_v) - 0.5 * w_bar * dot(β(x), β(x))  #here
    return ContinuousUnivariateLogPdf(log_backwardmess)
end