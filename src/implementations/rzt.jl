using LoopVectorization

"""
Calculates the Rayleigh statistic for a given time series.

# Arguments
- `t`: Time vector.
- `y`: Data vector.
- `dy`: Uncertainty vector (default is a vector of 1.0).
- `fstep`: Frequency step (default is NaN).
- `minfreq`: Minimum frequency (default is 2 / span of `t`).
- `maxfreq`: Maximum frequency (default is 10).
- `oversamp`: Oversampling factor (default value is sqrt(2)).

# Returns
- `Z`: Array of results (Z = n * R^2)
"""

function rayleigh(t::AbstractVector{<:Real},
                  y::AbstractVector{<:Real},
                  dy::AbstractVector{<:Real} = fill(1.0, length(t));
                  minfreq::Real = Float32(2 / (maximum(t) - minimum(t))),
                  maxfreq::Real = Float32(12),
                  oversamp::Real = Float32(5),
                  fstep::Real = Float32(2 / ((maximum(t) - minimum(t)) * oversamp)))

    minfreq::Float32, maxfreq::Float32, oversamp::Float32, fstep::Float32 = Float32(minfreq), Float32(maxfreq), Float32(oversamp), Float32(fstep)

    n::Int = length(t)

    x::Vector{Float32} = Float32.(t)         # convert t to float32 in order to maximize avx2 usage
    wy::Vector{Float32} = Float32.(y)        # convert y to float32 in order to maximize avx2 usage

    w::Vector{Float32} = Float32.(dy)
    w = 1 ./ (w)
    w ./= sum(w)         # normalize weights

    Y::Float32 = sum(w .* wy)    # calculate weighted average of weights

    wy = (y .- Y)

    wy .*= w

    Z_max::Float32 = sum(abs.(wy))        # length of sum of vectors can't be greater, than the sum of lengths of vectors

    ω, ωstep = 2 * π * minfreq,  2 * π * fstep # calculate angular frequencies

    cosdx, sindx = cos.(x .* ωstep), sin.(x .* ωstep)   # init values used for calculation of trigonometric recursions

    k::Int = cld((maxfreq - minfreq), fstep)
    Z::Vector{Float64} = Array{Float64}(undef, k) # Array of results - R = n * Z^2


#= # The commented loop contains code equivalent to the optimized loop below
    for i in 0:(k-1)
        local cosx::Vector{Float32}, sinx::Vector{Float32} = cos.((ω + (i * ωstep)) .* x), sin.((ω + (i * ωstep)) .* x)
        local C::Float32 = sum(cosx .* wy)
        local S::Float32 = sum(sinx .* wy)
        Z[i+1] = sqrt.((C*C + S*S) / Z_max)
    end
=#

    for i in 0:cld(k, 1024)
        local cosx::Vector{Float32}, sinx::Vector{Float32}, tmp = cos.((ω + (1024 * i * ωstep)) .* x), sin.((ω + (1024 * i * ωstep)) .* x), Array{Float32}(undef, n)

        for j in (1024 * i):(min(1024 * (i + 1) - 1, k - 1))
            local C::Float32, S::Float32 = 0, 0

            LoopVectorization.@avx for idx in 1:n
                C += cosx[idx] * wy[idx]
                S += sinx[idx] * wy[idx]
                tmp[idx]  = cosx[idx] * cosdx[idx] - sinx[idx] * sindx[idx]
                sinx[idx] = cosx[idx] * sindx[idx] + sinx[idx] * cosdx[idx]
                cosx[idx] = tmp[idx]
            end
            Z[j+1] = sqrt.((C*C + S*S) / Z_max)
        end
    end

    return Z
end
