@testsnippet test_snippet begin
    using RxGP
    using RxInfer
    using ReactiveMP
    using Random
    using Distributions
    using StableRNGs
    using KernelFunctions
    using LinearAlgebra
    using StatsFuns
    using Test

    const grad_default_method = ReactiveMP.ghcubature(21)

    function test_fixture(; D=1, fixed_input=false, fixed_output=false, kernel_spec=:SE, mode=:AN, independent_SE_lengthscales=true)
        rng = StableRNG(12)
        Nu = 5
        if D == 1
            q_in = fixed_input ? PointMass(rand(rng)) : NormalMeanVariance(rand(rng), rand(rng) + 0.1)
            q_out = fixed_output ? PointMass(rand(rng)) : NormalMeanVariance(rand(rng), rand(rng) + 0.1)
            q_Wg = PointMass(rand(rng) + 0.1)
            Xu = [rand(rng) for _ in 1:Nu]
            test_mean_fn = (x) -> x[] |> x -> x^2 + 0.35x
        elseif D > 1
            q_in = fixed_input ? PointMass(randn(rng, D)) : MvNormalMeanCovariance(randn(rng, D), randn(rng, D, D) |> x -> x * x' + 0.1I)
            q_out = fixed_output ? PointMass(randn(rng, D)) : MvNormalMeanCovariance(randn(rng, D), randn(rng, D, D) |> x -> x * x' + 0.1I)
            q_Wg = PointMass(randn(rng, D, D) |> x -> x * x' + 0.1I)
            Xu = [randn(rng, D) for _ in 1:Nu]
            test_mean_fn = (x) -> dot(x,x)
        end
        q_v = MvNormalMeanCovariance(randn(rng, Nu), randn(rng, Nu, Nu) |> x -> x * x' + 0.1I)
        kernel, θ_val, _ = get_simple_kernel_and_params(D; kernel_spec=kernel_spec, num_SE=2, num_SM=2, independent_SE_lengthscales=independent_SE_lengthscales)
        meta = get_UniSGPMeta(D; method=grad_default_method, mean_fn=test_mean_fn, kernel=kernel, kernel_spec=kernel_spec, mode=mode, operator=:grad, independent_SE_lengthscales=independent_SE_lengthscales, Xu=Xu, θ=θ_val)
        return (; method=grad_default_method, D, Xu, Nu=length(Xu), kernel, kernel_spec, mode, independent_SE_lengthscales, θ_val, meta, q_in, q_out, q_v, q_Wg, q_θ=PointMass(θ_val), rng)
    end
end

@testitem "helper_functions/other_functions/TestOtherFunction" setup=[test_snippet] begin
    using RxGP, Random, LinearAlgebra, Test
    A = rand(4,4)
    blk_A = [A[1:2,1:2], A[3:4,1:2], A[1:2,3:4],A[3:4,3:4]]
    a = rand(4)
    b = rand(4)

    @test jdotavx(a,b) ≈ dot(a,b)
    blk_matrix = create_blockmatrix(A,2,2)
    for i in eachindex(blk_matrix)
        @test blk_matrix[i] == blk_A[i]
    end
end

@testitem "helper_functions/other_functions/Test apply_mean_fn dispatches" setup=[test_snippet] begin
    mixed_mean(x) = x isa AbstractVector ? fill(sum(x), length(x)) : x^2

    scalar_input = 2.0
    scalar_result = apply_mean_fn(scalar_input, mixed_mean)
    @test scalar_result isa Number
    @test scalar_result == mixed_mean(scalar_input)

    length_one_vector = [3.0]
    length_one_result = apply_mean_fn(length_one_vector, mixed_mean)
    @test length_one_result isa Number
    @test length_one_result == mixed_mean(length_one_vector[1])

    multi_dim_vector = [1.0, 2.0, 3.0]
    multi_dim_result = apply_mean_fn(multi_dim_vector, mixed_mean)
    @test multi_dim_result isa AbstractVector
    @test size(multi_dim_result) == size(multi_dim_vector)

    nested_vectors = [[4.0], [5.0, 6.0]]
    nested_result = apply_mean_fn.(nested_vectors, Ref(mixed_mean))
    @test length(nested_result) == length(nested_vectors)
    @test nested_result[1] isa Number
    @test nested_result[2] isa AbstractVector
    @test length(nested_result[2]) == length(nested_vectors[2])
end

@testitem "helper_functions/other_functions/Test mean_cov_scalar_matrix" setup=[test_snippet] begin
    scalar_val = 1.5
    μ, Σ = mean_cov_scalar_matrix(scalar_val)
    @test μ == scalar_val
    @test size(Σ) == (1, 1)

    uni_dist = NormalMeanVariance(2.0, 0.3)
    μ, Σ = mean_cov_scalar_matrix(uni_dist)
    @test μ == mean(uni_dist)
    @test size(Σ) == (1, 1)
    @test Σ[1, 1] == var(uni_dist)

    scalar_pointmass = PointMass(4.2)
    μ, Σ = mean_cov_scalar_matrix(scalar_pointmass)
    @test μ == mean(scalar_pointmass)
    @test size(Σ) == (1, 1)

    vector_scalar = [7.0]
    μ, Σ = mean_cov_scalar_matrix(vector_scalar)
    @test μ == vector_scalar[1]
    @test size(Σ) == (1, 1)

    vector_with_dist = [NormalMeanVariance(-1.0, 0.2)]
    μ, Σ = mean_cov_scalar_matrix(vector_with_dist)
    @test μ == mean(vector_with_dist[1])
    @test size(Σ) == (1, 1)

    pointmass_vector = PointMass([9.0])
    μ, Σ = mean_cov_scalar_matrix(pointmass_vector)
    @test μ == 9.0
    @test size(Σ) == (1, 1)
end

@testitem "helper_functions/other_functions/Test mean_cov_vector_matrix" setup=[test_snippet] begin
    scalar_val = 0.5
    μ_vec, Σ = mean_cov_vector_matrix(scalar_val)
    @test length(μ_vec) == 1
    @test size(Σ) == (1, 1)

    uni_dist = NormalMeanVariance(-0.3, 1.2)
    μ_vec, Σ = mean_cov_vector_matrix(uni_dist)
    @test length(μ_vec) == 1
    @test Σ[1, 1] == var(uni_dist)

    cov2 = Matrix{Float64}(I, 2, 2)
    multi_dist = MvNormalMeanCovariance([1.0, -1.0], cov2)
    μ_vec, Σ = mean_cov_vector_matrix(multi_dist)
    @test length(μ_vec) == 2
    @test size(Σ) == (2, 2)

    real_vector = [2.0, 3.0, 4.0]
    μ_vec, Σ = mean_cov_vector_matrix(real_vector)
    @test μ_vec === real_vector
    @test size(Σ) == (3, 3)

    scalar_pointmass = PointMass(1.1)
    μ_vec, Σ = mean_cov_vector_matrix(scalar_pointmass)
    @test length(μ_vec) == 1
    @test size(Σ) == (1, 1)

    vector_pointmass = PointMass([1.0, 2.0])
    μ_vec, Σ = mean_cov_vector_matrix(vector_pointmass)
    @test length(μ_vec) == 2
    @test size(Σ) == (2, 2)

    wrapped_dist = [MvNormalMeanCovariance([0.0, 1.0], cov2)]
    μ_vec, Σ = mean_cov_vector_matrix(wrapped_dist)
    @test length(μ_vec) == 2
    @test size(Σ) == (2, 2)
end

@testitem "node_rule/univariate_grad/Test get_UniSGPMeta" setup=[test_snippet] begin
    ctx = test_fixture(D=3)
    meta = ctx.meta
    D = ctx.D
    θ = ctx.θ_val
    method = ctx.method
    mean_fn = getMeanFn(meta)
    kernel = ctx.kernel
    kernel_spec = ctx.kernel_spec
    mode = ctx.mode
    independent_SE_lengthscales = ctx.independent_SE_lengthscales
    Xu = ctx.Xu

    # composing meta as per get_UniSGPMeta
    dims_input = D
    Kuu = kernelmatrix(kernel(θ), Xu) + 1e-8 * I
    KuuF = cholesky(Kuu)
    x_dummy = zeros(D)
    Ψx = 0.0
    Ψxx = 0.0
    Ψ0 = kernelmatrix(kernel(θ), [x_dummy])[1]
    Ψ1_trans = kernelmatrix(kernel(θ), Xu, [x_dummy])
    Ψ2 = kernelmatrix(kernel(θ), Xu, [x_dummy]) * kernelmatrix(kernel(θ), [x_dummy], Xu);
    Ψ3 = kernelmatrix(kernel(θ), [x_dummy], Xu)
    Uv = zeros(size(Xu,1), size(Xu,1))

    Kxx_val = get_gradient_Kxx_fn(D; kernel=kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, θ)
    Kxu_val = get_gradient_Kxu_fn(D; kernel=kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, θ, Xu)

    dims_theta = length(θ)
    counter = 0
    N = 1

    @test getmethod(meta) == method
    @test getMeanFn(meta) == mean_fn
    @test getInducingInput(meta) == Xu
    @test getΨx(meta) == Ψx
    @test getΨxx(meta) == Ψxx
    @test getΨ0(meta) == Ψ0
    @test getΨ1_trans(meta) == Ψ1_trans
    @test getΨ2(meta) == Ψ2
    @test getΨ3(meta) == Ψ3
    @test length(getLm_fn(meta)(x_dummy)) == D
    @test getKxx_fn(meta)(x_dummy, θ) == Kxx_val
    @test getKxu_fn(meta)(x_dummy, θ, Xu) == Kxu_val
    @test getKuuF(meta) == KuuF
    @test getKernel(meta) == kernel
    @test get_dims_input(meta) == D
    @test get_dims_output(meta) == D # because grad operator
    @test get_dims_theta(meta) == dims_theta
    @test getUv(meta) == Uv
    @test getcounter(meta) == counter
    @test getN(meta) == N

end

@testitem "node_rule/univariate_grad/Test gradient mean function (Lm_fn)" setup=[test_snippet] begin
    function num_grad(x_dummy, mean_fn, D)
        ϵ = 1e-6
        numerical_grad = zeros(D)
        for i in 1:D
            x_plus = copy(x_dummy); x_minus = copy(x_dummy)
            x_plus[i] += ϵ; x_minus[i] -= ϵ
            numerical_grad[i] = (mean_fn(x_plus) - mean_fn(x_minus)) / (2 * ϵ)
        end
        return numerical_grad
    end

    D = 1
    mean_fn = (x) -> x[] |> x -> x^2 + 0.35x
    kernel, θ_val, _ = get_simple_kernel_and_params(D; kernel_spec=:SE)
    Xu = [[0.1 * i] for i in 1:5]
    meta = get_UniSGPMeta(D; method=grad_default_method, mean_fn=mean_fn, kernel=kernel, kernel_spec=:SE, mode=:AN, operator=:grad, Xu=Xu, θ=θ_val)
    Lm_fn = getLm_fn(meta)
    x_dummy = [1.3]
    # (a) Lm_fn is a function
    @test typeof(Lm_fn) <: Function
    # (b) produces vector output of correct dimension
    @test length(Lm_fn(x_dummy)) == D
    # (c) output is the gradient of mean_fn
    @test Lm_fn(x_dummy) ≈ num_grad(x_dummy, mean_fn, D) atol=1e-5
    @test Lm_fn(x_dummy) ≈ [2.95] atol=1e-10

    D = 3
    mean_fn = (x) -> sum(x .^ 2)
    kernel, θ_val, _ = get_simple_kernel_and_params(D; kernel_spec=:SE)
    Xu = [randn(D) for _ in 1:5]
    meta = get_UniSGPMeta(D; method=grad_default_method, mean_fn=mean_fn, kernel=kernel, kernel_spec=:SE, mode=:AN, operator=:grad, Xu=Xu, θ=θ_val)
    Lm_fn = getLm_fn(meta)
    x_dummy = [1.0, -2.0, 0.5]
    @test typeof(Lm_fn) <: Function
    @test length(Lm_fn(x_dummy)) == D
    @test Lm_fn(x_dummy) ≈ num_grad(x_dummy, mean_fn, D) atol=1e-5
    @test Lm_fn(x_dummy) ≈ [2.0, -4.0, 1.0] atol=1e-10
end

@testitem "node_rule/univariate_grad/Test get_gradient_Kxu_fn" setup=[test_snippet] begin
    function num_grad(x_dummy, ctx)
        f = (x) -> kernelmatrix(ctx.kernel(ctx.θ_val), [x], ctx.Xu)
        numerical_grad = zeros(ctx.D, length(ctx.Xu))
        ϵ = 1e-3
        for d in 1:ctx.D
            for j in 1:length(ctx.Xu)
                x_plus = copy(x_dummy)
                x_minus = copy(x_dummy)
                x_plus[d] += ϵ
                x_minus[d] -= ϵ
                numerical_grad[d, j] = (f(x_plus)[1,j] - f(x_minus)[1,j]) / (2 * ϵ)
            end
        end
        return numerical_grad
    end

    D, ctx, x_dummy = 1, test_fixture(D=1), [1.3]
    # Write test to confirm that get_gradient_Kxu_fn produces (a) an anonymous function 
    @test typeof(get_gradient_Kxu_fn(ctx.D; kernel=ctx.kernel, kernel_spec=ctx.kernel_spec, mode=ctx.mode, independent_SE_lengthscales=ctx.independent_SE_lengthscales)) <: Function
    # Write test to confirm that (b) this anonymous function produces matrix output of correct dimensionality
    Kxu_output = get_gradient_Kxu_fn(ctx.D; kernel=ctx.kernel, kernel_spec=ctx.kernel_spec, mode=ctx.mode, independent_SE_lengthscales=ctx.independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu)
    @test size(Kxu_output) == (ctx.D, length(ctx.Xu))

    GT_SE_1_true = [-0.2691768268281436 -0.5487623838673628 -0.46320326629878394 -0.40955812696807925 -0.2682954802768478]
    GT_SE_2_true = [-0.01086223379645479 -5.077773570641767e-5 -4.210768798680061e-6 0.02690647937092505 -0.28680713171080896; -0.08370736593113492 -6.99027597822954e-5 -1.0719550722848323e-5 -0.11035790396449197 -0.282397204787177]
    GT_SE_3_true = [-0.001780246166520165 -0.04500262834952751 -0.013001368731613283 -0.013707630908826873 -0.0068684743352934174; -0.006091754991503079 -0.09113439236249632 -0.02172433199824908 -0.01907013439609445 -0.020877724129803737; -0.0009669673239204313 -0.057592528454608605 -0.010689638904951004 0.0015008148252816655 0.0010888426857195987]

    GT_SE_1_false = [-0.2691768268281436 -0.5487623838673628 -0.46320326629878394 -0.40955812696807925 -0.2682954802768478]
    GT_SE_2_false = [-0.00989820535158073 -6.376312268677768e-5 -4.425331731739479e-6 0.02475194761961813 -0.2760314045595347; -0.08409690457808483 -9.677645518397249e-5 -1.2420552495035163e-5 -0.11192700810054232 -0.2996453581015912]
    GT_SE_3_false = [-0.0023479359035239334 -0.0464265107858533 -0.015424850027521454 -0.01784351622948887 -0.008395949354054154; -0.008588254286073446 -0.10050005249161512 -0.027550799025143242 -0.026535523313065898 -0.02728028012152859; -0.0014541024960974553 -0.06774409421589009 -0.014460122480166628 0.0022275222317791332 0.0015175803471760563]

    GT_SEn_1_true = [-0.592886566005757 -1.1704034117397955 -1.0097745501013167 -0.8965870690128552 -0.5909603650617878]
    GT_SEn_2_true = [-0.018567238760557048 -7.959017818632416e-5 -5.607037274764154e-6 0.04712571472872962 -0.5670910809549845; -0.15165112493189387 -0.00011618023847547468 -1.5137174749999852e-5 -0.20485530054157458 -0.5917155116627049]
    GT_SEn_3_true =  [-0.0033617704884650323 -0.08511673239374601 -0.024696761418438115 -0.027325459468426766 -0.013124095816427483; -0.011953142864624088 -0.179080083484634 -0.042876396386755736 -0.03949868743868823 -0.041449396466633655; -0.001968066136986274 -0.11737038654636989 -0.021882357489641166 0.0032241703142241174 0.0022421450476964658]

    GT_SEn_1_false = [-0.5648063109945367 -1.13407424374401 -0.9672272736661602 -0.8569009436470987 -0.5629637213466765]
    GT_SEn_2_false = [-0.01871071311958239 -9.981641467991326e-5 -6.5977909163065804e-6 0.04730362002001304 -0.5564300090573626; -0.15896953998165153 -0.00015149633422381468 -1.8517983217022572e-5 -0.21390449807261874 -0.6040315229244864]
    GT_SEn_3_false = [-0.004127634029038943 -0.0891354599542038 -0.028285766710401186 -0.03263542596178455 -0.015298977323217541; -0.015098015903736509 -0.19295265743630996 -0.05052207956838563 -0.04853293046865395 -0.04970973472823853; -0.002556289083762049 -0.1300636412655959 -0.026516669502669316 0.004074092469347895 0.002765312912166723]
    
    GT_SMn_1_true = [-5.505599865491761 2.166898399391992 -3.762707134255472 -5.114732125848842 -5.497641141519793]
    GT_SMn_2_true = [0.08649257924702239 3.313285104211796e-9 2.117638024188785e-9 -0.6206488657070353 -0.4590656093471368; 0.08745816292094442 3.3513373915292794e-9 2.141693909491255e-9 -0.6276248411027086 -0.46406290461895733]
    GT_SMn_3_true = [7.996008340688182e-5 -0.0026416667644390754 3.4380473269853936e-5 -0.002862808239446088 -0.004379520716238025; 8.058782735428393e-5 -0.0026621161301419163 3.464725559350037e-5 -0.0028849974998553744 -0.004414687348105518; 8.121557129054756e-5 -0.002682565496619555 3.491403795767394e-5 -0.0029071867607886014 -0.0044498539692146896]

    GT_SMn_1_false = [-5.505599865491761 2.166898399391992 -3.762707134255472 -5.114732125848842 -5.497641141519793]
    GT_SMn_2_false = [0.08649257924702239 3.313285104211796e-9 2.117638024188785e-9 -0.6206488657070353 -0.4590656093471368; 0.08745816292094442 3.3513373915292794e-9 2.141693909491255e-9 -0.6276248411027086 -0.46406290461895733]
    GT_SMn_3_false = [7.996008340688182e-5 -0.0026416667644390754 3.4380473269853936e-5 -0.002862808239446088 -0.004379520716238025; 8.058782735428393e-5 -0.0026621161301419163 3.464725559350037e-5 -0.0028849974998553744 -0.004414687348105518; 8.121557129054756e-5 -0.002682565496619555 3.491403795767394e-5 -0.0029071867607886014 -0.0044498539692146896]

    GT_SEn_SMn_1_true = [-6.398381514375995 1.1123763777458606 -4.902682593477259 -6.237760214811594 -6.3882682691882575]
    GT_SEn_SMn_2_true = [0.0643813894185899 -2.661563447826323e-5 -1.3262393067482056e-6 -0.6048154307965459 -0.9438481291474954; -0.03639073189515979 -3.984641731467544e-5 -3.6731051454156712e-6 -0.807428693580442 -0.9840322394722989]
    GT_SEn_SMn_3_true = [-0.0019929270778759785 -0.07046004725456967 -0.017360013980786265 -0.021474772442733533 -0.013088103980708234; -0.007347425615033881 -0.14892605747099386 -0.030717379722179465 -0.030703704685872135 -0.03324182826538579; -0.001190406891839104 -0.10003381277711582 -0.01595609892588671 0.00040536194648947005 -0.002436686218335385]
    
    GT_SEn_SMn_1_false = [-6.398381514375995 1.1123763777458606 -4.902682593477259 -6.237760214811594 -6.3882682691882575]
    GT_SEn_SMn_2_false = [0.0671896207670164 -2.2680163150554122e-5 -1.0465552340964774e-6 -0.602454451310519 -0.9466982277957781; -0.028772207572641192 -3.4423741459717475e-5 -2.9391917467742915e-6 -0.7972805393660354 -0.9945821877846706]
    GT_SEn_SMn_3_false = [-0.0017820056832228582 -0.06716078695052224 -0.01607850157016484 -0.020585583404792406 -0.012589556042406046; -0.006676453023260129 -0.14292032779091024 -0.028736485662899346 -0.029516822637570307 -0.031517966796392946; -0.0010801952833642235 -0.09704762042932191 -0.015071163344865593 -1.945860569289165e-5 -0.0027160849917487115]

    # Write tests to confirm that (c) this matrix output represents the gradient of (the kernelmatrix evaluated between x and Xu) with respect to input x at x_dummy
    # :SE Kernel
    kernel_spec, mode, independent_SE_lengthscales = :SE, :AD, true
    D, x_dummy, ctx = 1, [1.3], test_fixture(D=1, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SE_1_true atol=1e-5
    D, x_dummy, ctx = 2, [1.3, 2.7], test_fixture(D=2, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SE_2_true atol=1e-5
    D, x_dummy, ctx = 3, [1.3, 2.7, 0.5], test_fixture(D=3, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SE_3_true atol=1e-5

    kernel_spec, mode, independent_SE_lengthscales = :SE, :AN, true
    D, x_dummy, ctx = 1, [1.3], test_fixture(D=1, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SE_1_true atol=1e-5
    D, x_dummy, ctx = 2, [1.3, 2.7], test_fixture(D=2, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SE_2_true atol=1e-5
    D, x_dummy, ctx = 3, [1.3, 2.7, 0.5], test_fixture(D=3, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SE_3_true atol=1e-5

    kernel_spec, mode, independent_SE_lengthscales = :SE, :AD, false
    D, x_dummy, ctx = 1, [1.3], test_fixture(D=1, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SE_1_false atol=1e-5
    D, x_dummy, ctx = 2, [1.3, 2.7], test_fixture(D=2, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SE_2_false atol=1e-5
    D, x_dummy, ctx = 3, [1.3, 2.7, 0.5], test_fixture(D=3, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SE_3_false atol=1e-5

    kernel_spec, mode, independent_SE_lengthscales = :SE, :AN, false
    D, x_dummy, ctx = 1, [1.3], test_fixture(D=1, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SE_1_false atol=1e-5
    D, x_dummy, ctx = 2, [1.3, 2.7], test_fixture(D=2, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SE_2_false atol=1e-5
    D, x_dummy, ctx = 3, [1.3, 2.7, 0.5], test_fixture(D=3, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SE_3_false atol=1e-5

    # :SEn Kernel
    kernel_spec, mode, independent_SE_lengthscales = :SEn, :AD, true
    D, x_dummy, ctx = 1, [1.3], test_fixture(D=1, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SEn_1_true atol=1e-5
    D, x_dummy, ctx = 2, [1.3, 2.7], test_fixture(D=2, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SEn_2_true atol=1e-5
    D, x_dummy, ctx = 3, [1.3, 2.7, 0.5], test_fixture(D=3, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SEn_3_true atol=1e-5

    kernel_spec, mode, independent_SE_lengthscales = :SEn, :AN, true
    D, x_dummy, ctx = 1, [1.3], test_fixture(D=1, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SEn_1_true atol=1e-5
    D, x_dummy, ctx = 2, [1.3, 2.7], test_fixture(D=2, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SEn_2_true atol=1e-5
    D, x_dummy, ctx = 3, [1.3, 2.7, 0.5], test_fixture(D=3, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SEn_3_true atol=1e-5

    kernel_spec, mode, independent_SE_lengthscales = :SEn, :AD, false
    D, x_dummy, ctx = 1, [1.3], test_fixture(D=1, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SEn_1_false atol=1e-5
    D, x_dummy, ctx = 2, [1.3, 2.7], test_fixture(D=2, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SEn_2_false atol=1e-5
    D, x_dummy, ctx = 3, [1.3, 2.7, 0.5], test_fixture(D=3, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SEn_3_false atol=1e-5

    kernel_spec, mode, independent_SE_lengthscales = :SEn, :AN, false
    D, x_dummy, ctx = 1, [1.3], test_fixture(D=1, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SEn_1_false atol=1e-5
    D, x_dummy, ctx = 2, [1.3, 2.7], test_fixture(D=2, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SEn_2_false atol=1e-5
    D, x_dummy, ctx = 3, [1.3, 2.7, 0.5], test_fixture(D=3, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SEn_3_false atol=1e-5

    # :SMn Kernel
    kernel_spec, mode, independent_SE_lengthscales = :SMn, :AD, true
    D, x_dummy, ctx = 1, [1.3], test_fixture(D=1, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-4
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SMn_1_true atol=1e-4
    D, x_dummy, ctx = 2, [1.3, 2.7], test_fixture(D=2, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SMn_2_true atol=1e-5
    D, x_dummy, ctx = 3, [1.3, 2.7, 0.5], test_fixture(D=3, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SMn_3_true atol=1e-5
    
    kernel_spec, mode, independent_SE_lengthscales = :SMn, :AD, false
    D, x_dummy, ctx = 1, [1.3], test_fixture(D=1, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-4
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SMn_1_false atol=1e-4
    D, x_dummy, ctx = 2, [1.3, 2.7], test_fixture(D=2, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SMn_2_false atol=1e-5
    D, x_dummy, ctx = 3, [1.3, 2.7, 0.5], test_fixture(D=3, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SMn_3_false atol=1e-5

    # :SEn_SMn Kernel
    kernel_spec, mode, independent_SE_lengthscales = :SEn_SMn, :AD, true
    D, x_dummy, ctx = 1, [1.3], test_fixture(D=1, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-4
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SEn_SMn_1_true atol=1e-4
    D, x_dummy, ctx = 2, [1.3, 2.7], test_fixture(D=2, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SEn_SMn_2_true atol=1e-4
    D, x_dummy, ctx = 3, [1.3, 2.7, 0.5], test_fixture(D=3, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SEn_SMn_3_true atol=1e-4

    kernel_spec, mode, independent_SE_lengthscales = :SEn_SMn, :AD, false
    D, x_dummy, ctx = 1, [1.3], test_fixture(D=1, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-4
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SEn_SMn_1_false atol=1e-4
    D, x_dummy, ctx = 2, [1.3, 2.7], test_fixture(D=2, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SEn_SMn_2_false atol=1e-4
    D, x_dummy, ctx = 3, [1.3, 2.7, 0.5], test_fixture(D=3, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_grad(x_dummy, ctx) ≈ get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) atol=1e-5
    @test get_gradient_Kxu_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val, ctx.Xu) ≈ GT_SEn_SMn_3_false atol=1e-4

end

@testitem "node_rule/univariate_grad/Test get_gradient_Kxx_fn" setup=[test_snippet] begin
    function num_hessian(x_dummy, ctx)
        f = (x, x′) -> kernelmatrix(ctx.kernel(ctx.θ_val), [x], [x′])[1, 1]
        H = zeros(ctx.D, ctx.D)
        ϵ = 1e-3
        for i in 1:ctx.D, j in 1:ctx.D
            x_ip = copy(x_dummy); x_im = copy(x_dummy)
            x_jp = copy(x_dummy); x_jm = copy(x_dummy)
            x_ip[i] += ϵ; x_im[i] -= ϵ
            x_jp[j] += ϵ; x_jm[j] -= ϵ
            H[i, j] = ( f(x_ip, x_jp) - f(x_ip, x_jm) - f(x_im, x_jp) + f(x_im, x_jm) ) / (4ϵ^2)
        end
        return H
    end

    D, ctx, x_dummy = 1, test_fixture(D=1), [1.3]
    # Write test to confirm that get_gradient_Kxx_fn produces (a) an anonymous function 
    @test typeof(get_gradient_Kxx_fn(ctx.D; kernel=ctx.kernel, kernel_spec=ctx.kernel_spec, mode=ctx.mode, independent_SE_lengthscales=ctx.independent_SE_lengthscales)) <: Function
    # Write test to confirm that (b) this anonymous function produces matrix output of correct dimensionality
    Kxx_output = get_gradient_Kxx_fn(ctx.D; kernel=ctx.kernel, kernel_spec=ctx.kernel_spec, mode=ctx.mode, independent_SE_lengthscales=ctx.independent_SE_lengthscales)(x_dummy, ctx.θ_val)
    @test size(Kxx_output) == (ctx.D, ctx.D)

    GT_SE_1_true = [0.8616780045351473;;]
    GT_SE_2_true = [0.95 0.0; 0.0 0.8616780045351473]
    GT_SE_3_true = [0.9824762999138177 0.0 0.0; 0.0 0.919107766729374 0.0; 0.0 0.0 0.8616780045351473]

    GT_SE_1_false = [0.8616780045351473;;]
    GT_SE_2_false = [0.8616780045351473 0.0; 0.0 0.8616780045351473]
    GT_SE_3_false = [0.8616780045351473 0.0 0.0; 0.0 0.8616780045351473 0.0; 0.0 0.0 0.8616780045351473]

    GT_SEn_1_true = [1.9046229363461684;;]
    GT_SEn_2_true = [1.961693459254957 0.0; 0.0 1.8853886213193574]
    GT_SEn_3_true = [1.9870019032930195 0.0 0.0; 0.0 1.9309436608002581 0.0; 0.0 0.0 1.8772264944928927]

    GT_SEn_1_false = [1.811020087213123;;]
    GT_SEn_2_false = [1.811020087213123 0.0; 0.0 1.811020087213123]
    GT_SEn_3_false = [1.811020087213123 0.0 0.0; 0.0 1.811020087213123 0.0; 0.0 0.0 1.811020087213123]
    
    GT_SMn_1_true = [22.422523830120973;;]
    GT_SMn_2_true = [21.77759877264679 22.013892703539035; 22.013892703539035 22.25275120501692]
    GT_SMn_3_true = [21.532575059807037 21.69509273661524 21.857610413423444; 21.695092736615244 21.85883738742949 22.022582038243744; 21.857610413423444 22.022582038243744 22.18755366306404]

    GT_SMn_1_false = [22.422523830120973;;]
    GT_SMn_2_false = [21.77759877264679 22.013892703539035; 22.013892703539035 22.25275120501692]
    GT_SMn_3_false = [21.532575059807037 21.69509273661524 21.857610413423444; 21.695092736615244 21.85883738742949 22.022582038243744; 21.857610413423444 22.022582038243744 22.18755366306404]

    GT_SEn_SMn_1_true = [25.658634662337917;;]
    GT_SEn_SMn_2_true = [25.155338421694754 23.249315287928724; 23.249315287928724 25.42602019391631]
    GT_SEn_SMn_3_true = [24.942109631316477 22.983762743074095 23.0894469275934; 22.983762743074095 25.13372688663932 23.196107627700755; 23.089446927593393 23.196107627700755 25.326616926026535]
    
    GT_SEn_SMn_1_false = [25.658634662337917;;]
    GT_SEn_SMn_2_false = [24.84668888378917 22.96124535702411; 22.961245357024115 25.187736127935153]
    GT_SEn_SMn_3_false = [24.424951572812073 22.48622251115627 22.614361798774034; 22.48622251115627 24.68196467046993 22.74397013123646; 22.614361798774034 22.74397013123646 24.94044681297245]

    # Write tests to confirm that (c) this matrix output represents the gradient of (the kernelmatrix evaluated between x and Xu) with respect to input x at x_dummy
    # :SE Kernel
    kernel_spec, mode, independent_SE_lengthscales = :SE, :AD, true
    D, x_dummy, ctx = 1, [1.3], test_fixture(D=1, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-5
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SE_1_true atol=1e-5
    D, x_dummy, ctx = 2, [1.3, 2.7], test_fixture(D=2, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-5
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SE_2_true atol=1e-5
    D, x_dummy, ctx = 3, [1.3, 2.7, 0.5], test_fixture(D=3, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-5
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SE_3_true atol=1e-5

    kernel_spec, mode, independent_SE_lengthscales = :SE, :AN, true
    D, x_dummy, ctx = 1, [1.3], test_fixture(D=1, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-5
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SE_1_true atol=1e-5
    D, x_dummy, ctx = 2, [1.3, 2.7], test_fixture(D=2, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-5
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SE_2_true atol=1e-5
    D, x_dummy, ctx = 3, [1.3, 2.7, 0.5], test_fixture(D=3, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-5
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SE_3_true atol=1e-5

    kernel_spec, mode, independent_SE_lengthscales = :SE, :AD, false
    D, x_dummy, ctx = 1, [1.3], test_fixture(D=1, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-5
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SE_1_false atol=1e-5
    D, x_dummy, ctx = 2, [1.3, 2.7], test_fixture(D=2, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-5
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SE_2_false atol=1e-5
    D, x_dummy, ctx = 3, [1.3, 2.7, 0.5], test_fixture(D=3, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-5
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SE_3_false atol=1e-5

    kernel_spec, mode, independent_SE_lengthscales = :SE, :AN, false
    D, x_dummy, ctx = 1, [1.3], test_fixture(D=1, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-5
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SE_1_false atol=1e-5
    D, x_dummy, ctx = 2, [1.3, 2.7], test_fixture(D=2, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-5
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SE_2_false atol=1e-5
    D, x_dummy, ctx = 3, [1.3, 2.7, 0.5], test_fixture(D=3, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-5
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SE_3_false atol=1e-5

    # :SEn Kernel
    kernel_spec, mode, independent_SE_lengthscales = :SEn, :AD, true
    D, x_dummy, ctx = 1, [1.3], test_fixture(D=1, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-5
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SEn_1_true atol=1e-5
    D, x_dummy, ctx = 2, [1.3, 2.7], test_fixture(D=2, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-5
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SEn_2_true atol=1e-5
    D, x_dummy, ctx = 3, [1.3, 2.7, 0.5], test_fixture(D=3, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-5
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SEn_3_true atol=1e-5

    kernel_spec, mode, independent_SE_lengthscales = :SEn, :AN, true
    D, x_dummy, ctx = 1, [1.3], test_fixture(D=1, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-5
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SEn_1_true atol=1e-5
    D, x_dummy, ctx = 2, [1.3, 2.7], test_fixture(D=2, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-5
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SEn_2_true atol=1e-5
    D, x_dummy, ctx = 3, [1.3, 2.7, 0.5], test_fixture(D=3, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-5
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SEn_3_true atol=1e-5

    kernel_spec, mode, independent_SE_lengthscales = :SEn, :AD, false
    D, x_dummy, ctx = 1, [1.3], test_fixture(D=1, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-5
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SEn_1_false atol=1e-5
    D, x_dummy, ctx = 2, [1.3, 2.7], test_fixture(D=2, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-5
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SEn_2_false atol=1e-5
    D, x_dummy, ctx = 3, [1.3, 2.7, 0.5], test_fixture(D=3, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-5
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SEn_3_false atol=1e-5

    kernel_spec, mode, independent_SE_lengthscales = :SEn, :AN, false
    D, x_dummy, ctx = 1, [1.3], test_fixture(D=1, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-5
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SEn_1_false atol=1e-5
    D, x_dummy, ctx = 2, [1.3, 2.7], test_fixture(D=2, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-5
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SEn_2_false atol=1e-5
    D, x_dummy, ctx = 3, [1.3, 2.7, 0.5], test_fixture(D=3, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-5
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SEn_3_false atol=1e-5

    # :SMn Kernel
    kernel_spec, mode, independent_SE_lengthscales = :SMn, :AD, true
    D, x_dummy, ctx = 1, [1.3], test_fixture(D=1, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-3
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SMn_1_true atol=1e-5
    D, x_dummy, ctx = 2, [1.3, 2.7], test_fixture(D=2, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-3
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SMn_2_true atol=1e-5
    D, x_dummy, ctx = 3, [1.3, 2.7, 0.5], test_fixture(D=3, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-3
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SMn_3_true atol=1e-5
    
    kernel_spec, mode, independent_SE_lengthscales = :SMn, :AD, false
    D, x_dummy, ctx = 1, [1.3], test_fixture(D=1, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-3
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SMn_1_false atol=1e-5
    D, x_dummy, ctx = 2, [1.3, 2.7], test_fixture(D=2, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-3
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SMn_2_false atol=1e-5
    D, x_dummy, ctx = 3, [1.3, 2.7, 0.5], test_fixture(D=3, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-3
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SMn_3_false atol=1e-5

    # :SEn_SMn Kernel
    kernel_spec, mode, independent_SE_lengthscales = :SEn_SMn, :AD, true
    D, x_dummy, ctx = 1, [1.3], test_fixture(D=1, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-3
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SEn_SMn_1_true atol=1e-5
    D, x_dummy, ctx = 2, [1.3, 2.7], test_fixture(D=2, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-3
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SEn_SMn_2_true atol=1e-5
    D, x_dummy, ctx = 3, [1.3, 2.7, 0.5], test_fixture(D=3, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-3
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SEn_SMn_3_true atol=1e-5

    kernel_spec, mode, independent_SE_lengthscales = :SEn_SMn, :AD, false
    D, x_dummy, ctx = 1, [1.3], test_fixture(D=1, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-3
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SEn_SMn_1_false atol=1e-5
    D, x_dummy, ctx = 2, [1.3, 2.7], test_fixture(D=2, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-3
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SEn_SMn_2_false atol=1e-5
    D, x_dummy, ctx = 3, [1.3, 2.7, 0.5], test_fixture(D=3, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
    @test num_hessian(x_dummy, ctx) ≈ get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) atol=1e-3
    @test get_gradient_Kxx_fn(D; kernel=ctx.kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)(x_dummy, ctx.θ_val) ≈ GT_SEn_SMn_3_false atol=1e-5
end


@testitem "node_rule/univariate_grad/Test get_simple_kernel_and_params" setup=[test_snippet] begin

    GT_SE_1_true = [1.3132616875182228;;]
    GT_SE_2_true = [1.3132616875182228;;]
    GT_SE_3_true = [1.3132616875182228;;]

    GT_SE_1_false = [1.3132616875182228;;]
    GT_SE_2_false = [1.3132616875182228;;]
    GT_SE_3_false = [1.3132616875182228;;]

    GT_SEn_1_true = [3.1529363430707287;;]
    GT_SEn_2_true = [3.0971624284065618;;]
    GT_SEn_3_true = [3.0734478823784652;;]

    GT_SEn_1_false = [2.880557545929104;;]
    GT_SEn_2_false = [2.880557545929104;;]
    GT_SEn_3_false = [2.880557545929104;;]
    
    GT_SMn_1_true = [2.776544154856254;;]
    GT_SMn_2_true = [2.7089447806156057;;]
    GT_SMn_3_true = [2.68333340108536;;]

    GT_SEn_SMn_1_true = [6.196953130034116;;]
    GT_SEn_SMn_2_true = [6.0712058048927275;;]
    GT_SEn_SMn_3_true = [6.017800912766589;;]
    
    GT_SEn_SMn_1_false = [6.196953130034116;;]
    GT_SEn_SMn_2_false = [5.89641808371539;;]
    GT_SEn_SMn_3_false = [5.74077176619492;;]

    NSE, NSM = 2, 2 # 2, 2

    D, kernel_spec, independent_SE_lengthscales = 2, :SEn, false
    kernel, θ_val, dim_θ = get_simple_kernel_and_params(D; kernel_spec=kernel_spec, num_SE=NSE, num_SM=NSM, independent_SE_lengthscales=independent_SE_lengthscales)
    @test kernel isa Function
    @test length(θ_val) == dim_θ == 4

    θ_SE_1_true = collect(range(1, 2, length=2))
    θ_SE_2_true = collect(range(1, 2, length=3))
    θ_SE_3_true = collect(range(1, 2, length=4))

    θ_SE_1_false = collect(range(1, 2, length=2))
    θ_SE_2_false = collect(range(1, 2, length=2))
    θ_SE_3_false = collect(range(1, 2, length=2))
    
    θ_SEn_1_true = collect(range(1, 2, length=4))
    θ_SEn_2_true = collect(range(1, 2, length=6))
    θ_SEn_3_true = collect(range(1, 2, length=8))
    
    θ_SEn_1_false = collect(range(1, 2, length=4))
    θ_SEn_2_false = collect(range(1, 2, length=4))
    θ_SEn_3_false = collect(range(1, 2, length=4))

    θ_SMn_1 = collect(range(1, 2, length=6))
    θ_SMn_2 = collect(range(1, 2, length=10))
    θ_SMn_3 = collect(range(1, 2, length=14))
    
    θ_SEn_SMn_1_true = collect(range(1, 2, length=10))
    θ_SEn_SMn_2_true = collect(range(1, 2, length=16))
    θ_SEn_SMn_3_true = collect(range(1, 2, length=22))
    
    θ_SEn_SMn_1_false = collect(range(1, 2, length=10))
    θ_SEn_SMn_2_false = collect(range(1, 2, length=14))
    θ_SEn_SMn_3_false = collect(range(1, 2, length=18))
    
    x_dummy1 = [1.3]
    x_dummy2 = [1.3, 2.7]
    x_dummy3 = [1.3, 2.7, 0.5]

    @test kernelmatrix(get_simple_kernel_and_params(1; kernel_spec=:SE, num_SE=NSE, num_SM=NSM, independent_SE_lengthscales=true)[1](θ_SE_1_true), [x_dummy1]) ≈ GT_SE_1_true atol=1e-5
    @test kernelmatrix(get_simple_kernel_and_params(2; kernel_spec=:SE, num_SE=NSE, num_SM=NSM, independent_SE_lengthscales=true)[1](θ_SE_2_true), [x_dummy2]) ≈ GT_SE_2_true atol=1e-5
    @test kernelmatrix(get_simple_kernel_and_params(3; kernel_spec=:SE, num_SE=NSE, num_SM=NSM, independent_SE_lengthscales=true)[1](θ_SE_3_true), [x_dummy3]) ≈ GT_SE_3_true atol=1e-5
    @test kernelmatrix(get_simple_kernel_and_params(1; kernel_spec=:SE, num_SE=NSE, num_SM=NSM, independent_SE_lengthscales=false)[1](θ_SE_1_false), [x_dummy1]) ≈ GT_SE_1_false atol=1e-5
    @test kernelmatrix(get_simple_kernel_and_params(2; kernel_spec=:SE, num_SE=NSE, num_SM=NSM, independent_SE_lengthscales=false)[1](θ_SE_2_false), [x_dummy2]) ≈ GT_SE_2_false atol=1e-5
    @test kernelmatrix(get_simple_kernel_and_params(3; kernel_spec=:SE, num_SE=NSE, num_SM=NSM, independent_SE_lengthscales=false)[1](θ_SE_3_false), [x_dummy3]) ≈ GT_SE_3_false atol=1e-5
    
    @test kernelmatrix(get_simple_kernel_and_params(1; kernel_spec=:SEn, num_SE=NSE, num_SM=NSM, independent_SE_lengthscales=true)[1](θ_SEn_1_true), [x_dummy1]) ≈ GT_SEn_1_true atol=1e-5
    @test kernelmatrix(get_simple_kernel_and_params(2; kernel_spec=:SEn, num_SE=NSE, num_SM=NSM, independent_SE_lengthscales=true)[1](θ_SEn_2_true), [x_dummy2]) ≈ GT_SEn_2_true atol=1e-5
    @test kernelmatrix(get_simple_kernel_and_params(3; kernel_spec=:SEn, num_SE=NSE, num_SM=NSM, independent_SE_lengthscales=true)[1](θ_SEn_3_true), [x_dummy3]) ≈ GT_SEn_3_true atol=1e-5
    @test kernelmatrix(get_simple_kernel_and_params(1; kernel_spec=:SEn, num_SE=NSE, num_SM=NSM, independent_SE_lengthscales=false)[1](θ_SEn_1_false), [x_dummy1]) ≈ GT_SEn_1_false atol=1e-5
    @test kernelmatrix(get_simple_kernel_and_params(2; kernel_spec=:SEn, num_SE=NSE, num_SM=NSM, independent_SE_lengthscales=false)[1](θ_SEn_2_false), [x_dummy2]) ≈ GT_SEn_2_false atol=1e-5
    @test kernelmatrix(get_simple_kernel_and_params(3; kernel_spec=:SEn, num_SE=NSE, num_SM=NSM, independent_SE_lengthscales=false)[1](θ_SEn_3_false), [x_dummy3]) ≈ GT_SEn_3_false atol=1e-5
    
    @test kernelmatrix(get_simple_kernel_and_params(1; kernel_spec=:SMn, num_SE=NSE, num_SM=NSM, independent_SE_lengthscales=true)[1](θ_SMn_1), [x_dummy1]) ≈ GT_SMn_1_true atol=1e-5
    @test kernelmatrix(get_simple_kernel_and_params(2; kernel_spec=:SMn, num_SE=NSE, num_SM=NSM, independent_SE_lengthscales=true)[1](θ_SMn_2), [x_dummy2]) ≈ GT_SMn_2_true atol=1e-5
    @test kernelmatrix(get_simple_kernel_and_params(3; kernel_spec=:SMn, num_SE=NSE, num_SM=NSM, independent_SE_lengthscales=true)[1](θ_SMn_3), [x_dummy3]) ≈ GT_SMn_3_true atol=1e-5
    
    @test kernelmatrix(get_simple_kernel_and_params(1; kernel_spec=:SEn_SMn, num_SE=NSE, num_SM=NSM, independent_SE_lengthscales=true)[1](θ_SEn_SMn_1_true), [x_dummy1]) ≈ GT_SEn_SMn_1_true atol=1e-5
    @test kernelmatrix(get_simple_kernel_and_params(2; kernel_spec=:SEn_SMn, num_SE=NSE, num_SM=NSM, independent_SE_lengthscales=true)[1](θ_SEn_SMn_2_true), [x_dummy2]) ≈ GT_SEn_SMn_2_true atol=1e-5
    @test kernelmatrix(get_simple_kernel_and_params(3; kernel_spec=:SEn_SMn, num_SE=NSE, num_SM=NSM, independent_SE_lengthscales=true)[1](θ_SEn_SMn_3_true), [x_dummy3]) ≈ GT_SEn_SMn_3_true atol=1e-5
    @test kernelmatrix(get_simple_kernel_and_params(1; kernel_spec=:SEn_SMn, num_SE=NSE, num_SM=NSM, independent_SE_lengthscales=false)[1](θ_SEn_SMn_1_false), [x_dummy1]) ≈ GT_SEn_SMn_1_false atol=1e-5
    @test kernelmatrix(get_simple_kernel_and_params(2; kernel_spec=:SEn_SMn, num_SE=NSE, num_SM=NSM, independent_SE_lengthscales=false)[1](θ_SEn_SMn_2_false), [x_dummy2]) ≈ GT_SEn_SMn_2_false atol=1e-5
    @test kernelmatrix(get_simple_kernel_and_params(3; kernel_spec=:SEn_SMn, num_SE=NSE, num_SM=NSM, independent_SE_lengthscales=false)[1](θ_SEn_SMn_3_false), [x_dummy3]) ≈ GT_SEn_SMn_3_false atol=1e-5

end
