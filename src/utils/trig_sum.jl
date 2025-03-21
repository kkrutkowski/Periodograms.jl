using FFTW
# using LoopVectorization
# using StructArrays

# --- FFT Grid Helpers ---

function bitceil(N::Int)
    ispow2(N) && return N
    return 2^(floor(Int, log2(N)) + 1)
end

function next_fast_len(N::Int)
    Nfft = bitceil(N)
    if (Nfft ÷ 16) * 9 > N
        Nfft = (Nfft ÷ 16) * 9
    elseif (Nfft ÷ 128) * 81 > N
        Nfft = (Nfft ÷ 128) * 81
    elseif (Nfft ÷ 4) * 3 > N
        Nfft = (Nfft ÷ 4) * 3
    elseif (Nfft ÷ 32) * 27 > N
        Nfft = (Nfft ÷ 32) * 27
    end
    return Nfft
end

# --- Chebyshev Polynomials ---

function chebyshev_polynomials(d::Int, x::AbstractVector{T}) where T
    N = length(x)
    Tmat = zeros(T, N, d+1)
    @inbounds @simd for i in 1:N
        Tmat[i, 1] = one(T)
    end
    if d ≥ 1
        @inbounds @simd for i in 1:N
            Tmat[i, 2] = x[i]
        end
        two_x = 2 .* x
        for k in 2:d
            @inbounds @simd for i in 1:N
                Tmat[i, k+1] = two_x[i] * Tmat[i, k] - Tmat[i, k-1]
            end
        end
    end
    return Tmat
end

# --- Custom Bessel Function Approximation ---

"""
Approximate the Bessel function of the first kind (integer order) using a truncated series.
"""
function jn(n::Int, x::Real; num_terms::Int=10)
    if n < 0
        return (-1)^n * jn(-n, x; num_terms=num_terms)
    end
    half_x = 0.5 * x
    n_max = n + num_terms + 10
    fact = ones(Float64, n_max + 1)  # fact[1] is 0!
    @inbounds for i in 2:length(fact)
        fact[i] = fact[i-1] * (i - 1)
    end
    result = 0.0
    term_numer = half_x^n
    @inbounds for k in 0:(num_terms - 1)
        term_denom = fact[k+1] * fact[n + k + 1]
        result += term_numer / term_denom
        term_numer *= -half_x^2
    end
    return result
end

# --- Bessel Coefficients for NUFFT LRA Kernel ---

"""
    bessel_coefficients(K, γ)

Compute a K×K matrix of complex Bessel coefficients.
Only entries with even (q–p) are nonzero.
"""
function bessel_coefficients(K::Int, γ::T) where T<:Real
    cfs = zeros(Complex{T}, K, K)
    arg = -γ * π / 2
    @inbounds for p in 0:K-1, q in 0:K-1
        if iseven(q - p)
            cfs[p+1, q+1] = 4 * (1im)^q * jn((p+q) ÷ 2, arg; num_terms=10) * jn((q-p) ÷ 2, arg; num_terms=10)
        end
    end
    cfs[1, :] .*= 0.5
    cfs[:, 1] .*= 0.5
    return cfs
end

# --- Parameter Determination and Kernel Construction ---

"""
    get_lra_params(x, N, ϵ)

For nonuniform sample locations `x` and grid size `N`, compute:
  - `s`: wrapped grid indices (with 1-based indexing)
  - `γ`: maximum absolute deviation from the nearest grid point
  - `K`: truncation parameter (from a lookup table) for error tolerance `ϵ`
"""
function get_lra_params(x::AbstractVector{T}, N::Int, ϵ::T) where T<:Real
    M = length(x)
    s = mod.(round.(Int, N .* x), N) .+ 1
    er = mod.(N .* x - round.(N .* x) .+ 0.5, 1) .- 0.5
    γ = maximum(abs.(er))
    if γ ≤ ϵ
        K = 1
    else
        lut = Dict(2 => 2.6e-1,
                   3 => 5.4e-2,
                   4 => 8.6e-3,
                   5 => 1.1e-3,
                   6 => 1.3e-4,
                   7 => 1.2e-5,
                   8 => 1.2e-6,
                   9 => 9.0e-8,
                   10 => 6.8e-9,
                   11 => 4.3e-10,
                   12 => 2.9e-11,
                   13 => 1.6e-12,
                   14 => 8.4e-14,
                   15 => 2.2e-14,
                   16 => 4.6e-16)
        K = 16
        @inbounds for key in sort(collect(keys(lut)))
            if ϵ ≥ lut[key]
                K = key
                break
            end
        end
    end
    return s, γ, K
end

"""
    construct_UV(x, γ, K, N)

Construct the low-rank approximation matrices `U` and `V` so that the NUFFT
kernel is approximated by `U * Vᴴ`.
"""
function construct_UV(x::AbstractVector{T}, γ::T, K::Int, N::Int) where T<:Real
    M = length(x)
    er = mod.(N .* x - round.(N .* x) .+ 0.5, 1) .- 0.5
    Tcheb_U = γ == zero(T) ? ones(T, M, K) : chebyshev_polynomials(K-1, er ./ γ)
    B = bessel_coefficients(K, γ)
    U = exp.(-1im * π .* er) .* (Tcheb_U * B)
    ω = 0:N-1
    X = 2.0 .* ω ./ N .- 1.0
    V = complex.(chebyshev_polynomials(K-1, X))
    return U, V
end

# --- NUFFT Type-I via Low Rank Approximation ---

"""
    nufft1_lra(x, y, N, ϵ)

Compute the adjoint (Type-I) NUFFT using a low-rank approximation.
For nonuniform locations `x` (typically in [0,1)) and complex amplitudes `y`, computes

    s[k] = ∑ⱼ y[j] exp(2πi x[j]*(k-1))   for k = 1,…,N.
"""
function nufft1_lra(x::AbstractVector{T}, y::AbstractVector{T}, N::Int, ϵ::T) where T<:Real
    M = length(x)
    Nfft = next_fast_len(max(N, M))
    s_indices, γ, K = get_lra_params(x, Nfft, ϵ)
    U, V = construct_UV(x, γ, K, Nfft)
    temp = zeros(Complex{T}, Nfft, K)
    @inbounds for i in 1:M
        @inbounds for k in 1:K
            temp[s_indices[i], k] += conj(U[i, k]) * y[i]
        end
    end
    temp_ifft = FFTW.ifft(temp, 1) .* Nfft
    s = vec(sum(conj.(V) .* temp_ifft, dims=2))
    return s[1:N]
end

function nufft1_lra(x::AbstractVector{T}, y::AbstractVector{Complex{T}}, N::Int, ϵ::T) where T<:Real
    M = length(x)
    Nfft = next_fast_len(max(N, M))
    s_indices, γ, K = get_lra_params(x, Nfft, ϵ)
    U, V = construct_UV(x, γ, K, Nfft)
    temp = zeros(Complex{T}, Nfft, K)
    @inbounds for i in 1:M
        @inbounds for k in 1:K
            temp[s_indices[i], k] += conj(U[i, k]) * y[i]
        end
    end
    temp_ifft = FFTW.ifft(temp, 1) .* Nfft
    s = vec(sum(conj.(V) .* temp_ifft, dims=2))
    return s[1:N]
end

# --- The trig_sum Function ---

"""
    trig_sum(t, h, df, N; f0=0, freq_factor=1, oversampling=5,
             use_fft=true, Mfft=4, eps=5e-13)

Compute weighted sine and cosine sums for frequencies

    f[j] = freq_factor * (f0 + j * df)   for j = 0,…,N-1.

That is, compute

    S[j] = ∑ᵢ h[i] sin(2π f[j] t[i])
    C[j] = ∑ᵢ h[i] cos(2π f[j] t[i])

When using the FFT-based method, this is achieved via an approximate NUDFT.
"""
function trig_sum(t::Vector{T}, h::Vector{T}, df::T, N::Int;
                  f0::T=zero(T), freq_factor::T=one(T),
                  oversampling::Float64=5.0, use_fft::Bool=true,
                  eps::T=5e-13) where {T<:Real}
    if df <= 0
        error("df must be positive")
    end
    df *= freq_factor
    f0 *= freq_factor

    # Ensure h is complex if using FFT
    #if use_fft && !(T <: Complex)
    #    h = complex.(h)
    #end

    if use_fft
        t0 = minimum(t)

        if f0 != 0
            h = h .* exp.(2im * π * f0 .* (t .- t0))
        end
        tnorm = (t .- t0) .* df
        local fftgrid = nufft1_lra(tnorm, h, N, eps)
        if t0 > 0
            local f = f0 .+ df .* (0:N-1)
            fftgrid = fftgrid .* exp.(2im * π .* t0 .* f)
        end
        C = real.(fftgrid)
        S = imag.(fftgrid)

        #= elseif algorithm == "fasper"
            Nfft = next_fast_len(Int(N * oversampling))
            if f0 > 0
                h = h .* exp.(2im * π * f0 .* (t .- minimum(t)))
            end
            tnorm = mod.((t .- minimum(t)) .* Nfft .* df, Nfft)
            grid = extirpolate(tnorm, h, Nfft, Mfft)
            fftgrid = FFTW.ifft(grid) .* Nfft
            fftgrid = fftgrid[1:N]
            if t0 != 0
                f = f0 .+ df .* (0:N-1)
                fftgrid = fftgrid .* exp.(2im * π .* t0 .* f)
            end
            C = real.(fftgrid)
            S = imag.(fftgrid)
            else
            error("Unknown algorithm: $algorithm. Use 'fasper' or 'lra'.")
            end =#

    else
        dtmp_cos = cos.(2π * df .* t)
        dtmp_sin = sin.(2π * df .* t)

        block_size = 512
        nblocks = cld(N, block_size)

        C = zeros(T, N)
        S = zeros(T, N)

        tmp_cos = Vector{T}(undef, length(t))
        tmp_sin = Vector{T}(undef, length(t))

        C_tmp = zero(T)
        S_tmp = zero(T)
        tmp = zero(T)

        @inbounds for i in 0:(nblocks - 1)
            block_start = i * block_size + 1
            block_end   = min((i + 1) * block_size, N)

            f_start = f0 + df * (block_start - 1)

            tmp_cos = cos.(2π * f_start * t)
            tmp_sin = sin.(2π * f_start * t)

            @inbounds for j in block_start:block_end
                C_tmp = zero(T)
                S_tmp = zero(T)
                @inbounds @fastmath @simd for k in eachindex(t)
                    C_tmp += h[k] * tmp_cos[k]
                    S_tmp += h[k] * tmp_sin[k]

                    tmp = tmp_cos[k] * dtmp_cos[k] - tmp_sin[k] * dtmp_sin[k]
                    tmp_sin[k] = tmp_cos[k] * dtmp_sin[k] + tmp_sin[k] * dtmp_cos[k]
                    tmp_cos[k] = tmp
                end
                C[j] = C_tmp
                S[j] = S_tmp
            end
        end
    # the loop above is mathematically equivalent to
    # dftgrid = [sum(h .* exp.(2π * 1im * f_j .* t)) for f_j in f]
    end
    return S, C
end

#= --- extirpolate Helper (for 'fasper' algorithm) ---

"""
    extirpolate(x, y, N, M)

Extirpolate the values `(x, y)` onto an integer grid `0:N-1` using Lagrange
polynomial weights on the `M` nearest points.
"""

# function extirpolate(x::Vector{T}, y::Vector{T2}, N::Int, M::Int) where {T<:Real, T2<:Number}
#     result = zeros(T2, N)
#     # First add contributions for integer-valued x
#     for (xi, yi) in zip(x, y)
#         if isapprox(mod(xi,1), 0; atol=1e-12)
#             idx = Int(round(xi)) + 1
#             if 1 ≤ idx ≤ N
#                 result[idx] += yi
#             end
#         end
#     end
#     # Process non-integer x
#     for (xi, yi) in zip(x, y)
#         if !isapprox(mod(xi,1), 0; atol=1e-12)
#             ilo = clamp(floor(Int, xi - div(M,2)), 0, N - M)
#             num = yi * prod(xi - (ilo + j) for j in 0:M-1)
#             den = factorial(M - 1)
#             for j in 0:M-1
#                 if j > 0
#                     den *= j / (j - M)
#                 end
#                 ind = ilo + (M - 1 - j)
#                 result[ind + 1] += num / (den * (xi - ind))
#             end
#         end
#     end
#     return result
# end

=#
