using Periodograms
using Test, Random, Statistics

function test_trig_sum(use_fft::Bool)
    Random.seed!(1234)
    M = 400         # number of nonuniform sample points
    N = 2048        # number of frequency bins
    ϵ = 1e-15       # tolerance
    t = sort(rand(M))
    h = randn(M) .* randn(M)
    df = 0.1        # frequency spacing
    f0 = 0.05

    # Use FFT-based method with the low-rank approximation ("lra")
    S_fft, C_fft = Periodograms.trig_sum(t, h, df, N; f0=f0, use_fft=use_fft, eps=ϵ)

    # Compute direct summation using the complex exponential (the proper comparison)
    f = f0 .+ df .* (0:N-1)
    s_direct = [sum(h .* exp.(2π * 1im * f_j .* t)) for f_j in f]
    C_direct = real.(s_direct)
    S_direct = imag.(s_direct)

    rel_err_C = abs.(C_direct .- C_fft) ./ (abs.(C_direct) .+ ϵ)
    rel_err_S = abs.(S_direct .- S_fft) ./ (abs.(S_direct) .+ ϵ)

    println("Median relative error (cosine): ", median(rel_err_C))
    println("Maximum relative error (cosine): ", maximum(rel_err_C))
    println("Median relative error (sine): ", median(rel_err_S))
    println("Maximum relative error (sine): ", maximum(rel_err_S))

    if maximum(rel_err_C) > 1e-9 || maximum(rel_err_S) > 1e-9
        return 1
    else
        return 0
    end
end

@testset "Periodograms.jl" begin
    # Write your tests here.
    @test Periodograms.greet_your_package_name() == "Hello Periodograms!"
    @test Periodograms.greet_your_package_name() != "Hello world!"
    @test test_trig_sum(true) == 0
    @test test_trig_sum(false) == 0
end
