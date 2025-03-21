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

    # Input validation
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

    # Weighting and data centering
    w = 1.0 ./(dy .^ 2)
    ws = sum(w)
    y .-= (sum(w .* y) / ws)
    chi2_ref = sum((y ./ dy) .^ 2)

    # Precompute trig sums. Note that trig_sum is assumed to return two vectors.
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

    # Preallocate buffers for the linear system.
    XTX = zeros(norder, norder)
    XTy = zeros(norder)
    p = similar(zeros(Nf))

    # Loop over frequency grid
    @inbounds for i in 1:Nf
        # Fill cosine-cosine block and corresponding XTy entries.
        @inbounds for a in 0:nterms
            ia = a + 1
            # The cosine block: for b from a to nterms.
            @inbounds for b in a:nterms
                jb = b + 1
                XTX_val = 0.5 * (Cw[abs(a - b) + 1][i] + Cw[a + b + 1][i])
                XTX[ia, jb] = XTX_val
                XTX[jb, ia] = XTX_val
            end
            # Fill XTy for cosine terms.
            XTy[ia] = Cyw[a + 1][i]
        end

        # Fill sine-sine block and corresponding XTy entries.
        @inbounds for a in 1:nterms
            ia = nC + a
            @inbounds for b in a:nterms
                jb = nC + b
                XTX_val = 0.5 * (Cw[abs(a - b) + 1][i] - Cw[a + b + 1][i])
                XTX[ia, jb] = XTX_val
                XTX[jb, ia] = XTX_val
            end
            XTy[ia] = Syw[a + 1][i]
        end

        # Fill cosine-sine cross terms.
        @inbounds for a in 0:nterms
            ia = a + 1
            @inbounds for b in 1:nterms
                jb = nC + b
                s = b - a  # sign(s) will be -1, 0, or 1.
                XTX_val = 0.5 * (sign(s) * Sw[abs(s) + 1][i] + Sw[a + b + 1][i])
                XTX[ia, jb] = XTX_val
                XTX[jb, ia] = XTX_val  # symmetry.
            end
        end

        # Solve the linear system XTX * β = XTy.
        # We assume that solve() efficiently uses a Cholesky (LLT) decomposition.
        β = solve(XTX, XTy)
        p[i] = sum(XTy .* β)
    end

    return normalization == "none" ? p ./ chi2_ref : error("Normalization '$normalization' not recognized")
end
