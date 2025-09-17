using Statistics
using StaticArrays

# Assumes the following files and the vectorized `solve` function are available.
include("../utils/trig_sum.jl")
include("../utils/mle.jl")
# include("path/to/vectorized_solve.jl") # <-- Include your new solve function here

# Define a type alias for an 8-element vector of Float64.
const Vec8 = SVector{8, Float64}

"""
    fastchi2(t, y, dy, f0, df, Nf; ...)

Compute the fast chi² periodogram, solving 8 frequencies at a time.
(Function signature and arguments remain unchanged).
"""
function fastchi2(t::Vector{Float64}, y::Vector{Float64}, dy::Vector{Float64},
                  f0::Float64, df::Float64, Nf::Int;
                  normalization::String="none",
                  nterms::Int=3,
                  use_fft::Union{Bool, Symbol} = :auto,
                  eps::Float64=5e-13)

    # --- Input validation and setup (unchanged) ---
    if nterms <= 0
        error("Cannot have nterms <= 0")
    end
    if f0 < 0 || df <= 0 || Nf <= 0
        error("Invalid frequency parameters")
    end
    N = length(t)
    if length(y) != N || length(dy) != N
        error("t, y, and dy must all have the same length")
    end

    w = 1.0 ./(dy .^ 2)
    ws = sum(w)
    y .-= (sum(w .* y) / ws)
    chi2_ref = sum((y ./ dy) .^ 2)

    # --- Precompute trig sums (unchanged) ---
    Cw = Vector{Vector{Float64}}(undef, 2 * nterms + 1)
    Sw = Vector{Vector{Float64}}(undef, 2 * nterms + 1)
    Cyw = Vector{Vector{Float64}}(undef, nterms + 1)
    Syw = Vector{Vector{Float64}}(undef, nterms + 1)

    Cw[1] = ws * ones(Nf)
    Sw[1] = zeros(Nf)
    Syw[1] = sum(y .* w) * ones(Nf)
    Cyw[1] = zeros(Nf)

    @inbounds for i in 1:(2 * nterms)
        Sw[i + 1], Cw[i + 1] = trig_sum(t, w, df, Nf; f0=f0, freq_factor=Float64(i), eps=eps)
    end
    @inbounds for i in 1:nterms
        Syw[i + 1], Cyw[i + 1] = trig_sum(t, w .* y, df, Nf; f0=f0, freq_factor=Float64(i), eps=eps)
    end

    nC = nterms + 1
    nS = nterms
    norder = nC + nS
    p = zeros(Nf)

    # --- Vectorized Main Loop ---
    # Process frequencies in vectorized chunks of 8.
    Nf_vec = Nf - (Nf % 8)

    @inbounds for i in 1:8:Nf_vec
        freq_slice = i:(i + 7)
        XTX_vec = zeros(Vec8, norder, norder)
        XTy_vec = zeros(Vec8, norder)

        # Fill cosine-cosine block and corresponding XTy entries
        @inbounds for a in 0:nterms
            ia = a + 1
            @inbounds for b in a:nterms
                jb = b + 1
                c1 = Vec8(@view Cw[abs(a - b) + 1][freq_slice])
                c2 = Vec8(@view Cw[a + b + 1][freq_slice])
                XTX_val_vec = 0.5 * (c1 + c2)
                XTX_vec[ia, jb] = XTX_val_vec
                XTX_vec[jb, ia] = XTX_val_vec
            end
            XTy_vec[ia] = Vec8(@view Cyw[a + 1][freq_slice])
        end

        # Fill sine-sine block and corresponding XTy entries
        @inbounds for a in 1:nterms
            ia = nC + a
            @inbounds for b in a:nterms
                jb = nC + b
                c1 = Vec8(@view Cw[abs(a - b) + 1][freq_slice])
                c2 = Vec8(@view Cw[a + b + 1][freq_slice])
                XTX_val_vec = 0.5 * (c1 - c2)
                XTX_vec[ia, jb] = XTX_val_vec
                XTX_vec[jb, ia] = XTX_val_vec
            end
            XTy_vec[ia] = Vec8(@view Syw[a + 1][freq_slice])
        end

        # Fill cosine-sine cross terms
        @inbounds for a in 0:nterms
            ia = a + 1
            @inbounds for b in 1:nterms
                jb = nC + b
                s = b - a
                s1 = Vec8(@view Sw[abs(s) + 1][freq_slice])
                s2 = Vec8(@view Sw[a + b + 1][freq_slice])
                XTX_val_vec = 0.5 * (sign(s) * s1 + s2)
                XTX_vec[ia, jb] = XTX_val_vec
                XTX_vec[jb, ia] = XTX_val_vec
            end
        end

        # Solve the 8 linear systems simultaneously
        β_vec = solve(XTX_vec, XTy_vec)

        # Calculate and store the power for the 8 systems
        p[freq_slice] = sum(XTy_vec .* β_vec)
    end

    # --- Scalar Remainder Loop ---
    # Process any remaining frequencies one-by-one.
    if Nf_vec < Nf
        XTX = zeros(norder, norder)
        XTy = zeros(norder)
        @inbounds for i in (Nf_vec + 1):Nf
            # Fill cosine-cosine block
            @inbounds for a in 0:nterms, b in a:nterms
                XTX_val = 0.5 * (Cw[abs(a - b) + 1][i] + Cw[a + b + 1][i])
                XTX[a + 1, b + 1] = XTX[b + 1, a + 1] = XTX_val
            end
            @inbounds for a in 0:nterms
                XTy[a + 1] = Cyw[a + 1][i]
            end

            # Fill sine-sine block
            @inbounds for a in 1:nterms, b in a:nterms
                XTX_val = 0.5 * (Cw[abs(a - b) + 1][i] - Cw[a + b + 1][i])
                XTX[nC + a, nC + b] = XTX[nC + b, nC + a] = XTX_val
            end
            @inbounds for a in 1:nterms
                XTy[nC + a] = Syw[a + 1][i]
            end

            # Fill cosine-sine cross terms
            @inbounds for a in 0:nterms, b in 1:nterms
                s = b - a
                XTX_val = 0.5 * (sign(s) * Sw[abs(s) + 1][i] + Sw[a + b + 1][i])
                XTX[a + 1, nC + b] = XTX[nC + b, a + 1] = XTX_val
            end

            # Solve the single linear system
            β = XTX \ XTy
            p[i] = sum(XTy .* β)
        end
    end

    # --- Normalization and return (unchanged) ---
    return normalization == "none" ? p ./ chi2_ref : error("Normalization '$normalization' not recognized")
end
