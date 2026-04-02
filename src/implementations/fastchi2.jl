using LinearAlgebra
using Statistics

include("./trig_sum.jl")
include("./mle.jl")

# Ensure the type alias from the previous step is available
const CVec8 = Vector{ComplexF64}

"""
fastchi2(t, y, dy, f0, df, Nf; ...)

Compute the fast chi² periodogram using a pure Complex Toeplitz formulation.
"""
function fastchi2(t::Vector{Float64}, y::Vector{Float64}, dy::Vector{Float64},
                          f0::Float64, df::Float64, Nf::Int;
                          normalization::String="none",
                          nterms::Int=3,
                          eps::Float64=5e-13)

    if nterms <= 0 error("Cannot have nterms <= 0") end
    if f0 < 0 || df <= 0 || Nf <= 0 error("Invalid frequency parameters") end

    N = length(t)
    if length(y) != N || length(dy) != N error("Lengths of t, y, dy must match") end

    # Weight setup
    w = 1.0 ./ (dy .^ 2)
    ws = sum(w)
    y .-= (sum(w .* y) / ws)
    chi2_ref = sum((y ./ dy) .^ 2)

    # Precompute trig sums
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

    norder = 2 * nterms + 1
    p = zeros(Float64, Nf)

    # --- Batched Levinson Workspace ---
    R_batch      = [zeros(ComplexF64, 8) for _ in 1:norder]
    Y_batch      = [zeros(ComplexF64, 8) for _ in 1:norder]
    beta_batch   = [zeros(ComplexF64, 8) for _ in 1:norder]
    a_batch      = [zeros(ComplexF64, 8) for _ in 1:norder]
    a_prev_batch = [zeros(ComplexF64, 8) for _ in 1:norder]

    # Loop over frequencies in chunks of 8
    @inbounds for freq_base in 1:8:Nf

        # Fill the batch (1 to 8 systems)
        for sys in 1:8
            freq_idx = freq_base + sys - 1
            # Boundary handling: if we overshoot Nf, duplicate the last valid frequency
            valid_idx = freq_idx <= Nf ? freq_idx : Nf

            for k in 1:norder
                R_batch[k][sys] = Cw[k][valid_idx] - im * Sw[k][valid_idx]

                h = k - nterms - 1
                if h == 0
                    Y_batch[k][sys] = Cyw[1][valid_idx] + im * 0.0
                elseif h > 0
                    Y_batch[k][sys] = Cyw[h + 1][valid_idx] - im * Syw[h + 1][valid_idx]
                else
                    Y_batch[k][sys] = Cyw[abs(h) + 1][valid_idx] + im * Syw[abs(h) + 1][valid_idx]
                end
            end
        end

        # Solve the 8 systems simultaneously using the vectorized Levinson-Durbin
        solve_levinson!(beta_batch, R_batch, Y_batch, a_batch, a_prev_batch)

        # Extract the power
        for sys in 1:8
            freq_idx = freq_base + sys - 1
            if freq_idx <= Nf
                dot_product = zero(ComplexF64)
                for k in 1:norder
                    dot_product += conj(beta_batch[k][sys]) * Y_batch[k][sys]
                end
                p[freq_idx] = real(dot_product)
            end
        end
    end

return normalization == "none" ? p ./ chi2_ref : error("Normalization '$normalization' not recognized")
end
