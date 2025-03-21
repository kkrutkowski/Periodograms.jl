using Statistics

include("../utils/trig_sum.jl")
include("../utils/mle.jl")


"""
    fastchi2(t, y, dy, f0, df, Nf;
                         normalization="standard",
                         nterms=1, use_fft=true,
                         eps=5e-13)

Compute the fast chi² periodogram (equivalent to the Astropy Lomb–Scargle periodogram).

# Arguments
- `t`, `y`, `dy` : vectors of observation times, data values, and errors.
- `f0`, `df`, `Nf` : frequency grid parameters. Frequencies are given by
  `f = f0 + df * (0:Nf-1)`.
- `normalization` : one of `"standard"`, `"none"`.
- `nterms` : number of Fourier terms in the fit.
- `use_fft` : if true, use the FFT-based method in `trig_sum`.

# Returns
A vector of periodogram power values at each frequency.
"""
function fastchi2(t::Vector{Float64}, y::Vector{Float64}, dy::Vector{Float64},
                              f0::Float64, df::Float64, Nf::Int;
                              normalization::String="none",
                              nterms::Int=3,
                              use_fft::Union{Bool, Symbol} = :auto,
                              eps::Float64=5e-13)

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
    if center_data || fit_mean
        y = y .- (sum(w .* y) / ws)
    end

    chi2_ref = sum((y ./ dy) .^ 2)

    # Compute necessary sums
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

    # Construct the order of terms
    order = Tuple{Char, Int}[]
    push!(order, ('C', 0))
    @inbounds for i in 1:nterms
        push!(order, ('S', i))
        push!(order, ('C', i))
    end

    function compute_power(i::Int)
        norder = length(order)
        XTX = zeros(norder, norder)
        XTy = zeros(norder)

        @inbounds for j in 1:norder
            A = order[j]
            @inbounds for k in 1:norder
                B = order[k]
                if A[1] == 'S' && B[1] == 'S'
                    XTX[j, k] = 0.5 * (Cw[abs(A[2] - B[2]) + 1][i] - Cw[A[2] + B[2] + 1][i])
                elseif A[1] == 'C' && B[1] == 'C'
                    XTX[j, k] = 0.5 * (Cw[abs(A[2] - B[2]) + 1][i] + Cw[A[2] + B[2] + 1][i])
                elseif A[1] == 'S' && B[1] == 'C'
                    XTX[j, k] = 0.5 * (sign(A[2] - B[2]) * Sw[abs(A[2] - B[2]) + 1][i] + Sw[A[2] + B[2] + 1][i])
                elseif A[1] == 'C' && B[1] == 'S'
                    XTX[j, k] = 0.5 * (sign(B[2] - A[2]) * Sw[abs(B[2] - A[2]) + 1][i] + Sw[B[2] + A[2] + 1][i])
                else
                    XTX[j, k] = Cw[1][i]  # Bias term
                end
            end
            if A[1] == 'S'
                XTy[j] = Syw[A[2] + 1][i]
            else
                XTy[j] = Cyw[A[2] + 1][i]
            end
        end

        β = solve(XTX, XTy)
        return sum(XTy .* β)
    end

    p = [compute_power(i) for i in 1:Nf]
    return normalization == "none" ? p ./ chi2_ref : error("Normalization '$normalization' not recognized")
end
