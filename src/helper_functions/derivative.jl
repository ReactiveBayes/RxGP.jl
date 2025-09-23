# Define gradient functions for hyperparameter optimization
using ForwardDiff, KernelFunctions
using LinearAlgebra

export neg_log_backwardmess_fast, neg_log_backwardmess_uncertain, grad_llh_new_default!, grad_llh_new!, grad_llh_uncertain!
export neg_log_backwardmess_multi, grad_llh_multi, grad_llh_multi!

## The following functions are copied from the backward message toward "θ" edge
# 1) univariate case
#for regression/classification with known input 
function neg_log_backwardmess_fast(θ; y_data, x_data, v, Uv, w, kernel, Xu)
    Kuu = kernelmatrix(kernel(θ), Xu) + 1e-12*I
    Lu = fastcholesky(Kuu).L
    kxx = kernelmatrix_diag(kernel(θ), x_data)
    Kux = kernelmatrix(kernel(θ), Xu, x_data)
    
    llh = 0.0
    α = Lu \ view(Kux, :, 1)
    β = Uv * view(Kux, :, 1)
    llh += -0.5 * w * view(kxx, 1) .+ 0.5 * w * dot(α, α) .- 0.5 * w * dot(β, β) .+ w * view(y_data,1) * dot(v, view(Kux, :, 1))
    @inbounds @simd for i in 2:size(y_data,1)
        ldiv!(α, Lu , view(Kux, :, i))
        mul!(β,Uv, view(Kux, :, i))
        llh += -0.5 * w * view(kxx, i) .+ 0.5 * w * dot(α, α) .- 0.5 * w * dot(β, β) .+ w * view(y_data,i) * dot(v, view(Kux, :, i))
    end
    return -llh
end

#case that input is random variable 
function neg_log_backwardmess_uncertain(θ; y_data, qx,v, Uv,w,kernel,Xu,method)
    Kuu_inverse = inv(kernelmatrix(kernel(θ), Xu) + 1e-12*I) 
    llh = 0.0
    @inbounds for i in eachindex(y_data)
        Ψ0 = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ), [x], [x]),qx[i])[1]
        Ψ1_trans = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ), Xu,[x]),qx[i]) 
        Ψ2 = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), qx[i])
        llh += w * y_data[i] * dot(Ψ1_trans, v) - 0.5 * w * (Ψ0 + tr(Ψ2*(Uv'*Uv - Kuu_inverse)))
    end
    return -llh
end

## Define the corresponding derivative (gradient) functions 
function grad_llh_new_default!(grad,θ; y_data,x_data,v,Uv,w,kernel,Xu)
    return ForwardDiff.gradient!(grad, (x) -> neg_log_backwardmess_fast(x;y_data = y_data,x_data=x_data,v=v,Uv=Uv,w=w,kernel=kernel,Xu=Xu), θ)
end

#this is for big data
function grad_llh_new!(grad, θ; y_data,x_data,v,Uv,w,kernel,Xu,chunk_size)
    newfunc = (x) -> neg_log_backwardmess_fast(x;y_data = y_data,x_data=x_data,v=v,Uv=Uv,w=w,kernel=kernel,Xu=Xu)
    cfg = GradientConfig(newfunc, θ, Chunk{chunk_size}())
    return ForwardDiff.gradient!(grad, newfunc, θ,cfg)
end

function grad_llh_uncertain!(grad,θ; y_data, qx,v, Uv,w,kernel,Xu,method)
    return ForwardDiff.gradient!(grad, (x) -> neg_log_backwardmess_uncertain(x; y_data=y_data, qx=qx,v=v, Uv=Uv,w=w,kernel=kernel,Xu=Xu,method=method), θ)
end

# 2) multivariate case
"""
this function is only for C = I
    y_data: Array of mean.(qy), or array of output 
    qx : Array of distributions 
    sumRv_Wbar : Real, sum(Rv_blk .* W)
    v : mean(qv)
    W : mean(qW)
    Tr_W : trace(W)
    kernel : kernel of gp 
    Xu : inducing points 
    method: method to approximate kernel expectation 
"""
function neg_log_backwardmess_multi(θ;y_data, qx, sumRv_Wbar, v, W ,tr_W, kernel, Xu,method)
        Kuu_inverse = inv(kernelmatrix(kernel(θ), Xu) + 1e-12*I)        
        llh = 0.0
        M = size(Xu,1)
        @inbounds for i in eachindex(qx)
            @inbounds V = v * y_data[i]' * W
            sumdiagV = sum_diagonal_M(V,M)
            @inbounds Ψ_0 = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ), [x], [x]),qx[i])[1] # kxx
            @inbounds Ψ_1_trans = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ), Xu,[x]),qx[i]) # kux
            @inbounds Ψ_2 = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), qx[i]) # kux*kxu 

            llh += -0.5 * tr_W * (Ψ_0 - sum(Kuu_inverse .* Ψ_2)) + sum(sumdiagV .* Ψ_1_trans) - 0.5 * sum(sumRv_Wbar .* Ψ_2)
        end
    return -llh
end

function grad_llh_multi(θ;y_data, qx, sumRv_Wbar, v, W ,tr_W, kernel, Xu,method)
    return ForwardDiff.gradient((x) -> neg_log_backwardmess_multi(x;y_data, qx, sumRv_Wbar, v, W ,tr_W, kernel, Xu,method), θ)
end

function grad_llh_multi!(grad,θ;y_data, qx, sumRv_Wbar, v, W ,tr_W, kernel, Xu,method)
    return ForwardDiff.gradient!(grad,(x) -> neg_log_backwardmess_multi(x;y_data, qx, sumRv_Wbar, v, W ,tr_W, kernel, Xu,method), θ)
end

function sum_diagonal_M(V, M)
    @inbounds sumΣV = sum(view(V,M*(i-1)+1:i*M,i)  for i=1:size(V,2))
    return sumΣV
end

#compute trace of block matrix 
function trace_blkmatrix(Rv,D,M)
    return [tr(view(Rv,i:i+M-1,j:j+M-1)) for i=1:M:M*D, j=1:M:M*D]
end