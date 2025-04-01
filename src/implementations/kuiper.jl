# Kuiper's test approximate implementation with two methods:
# "fast" uses a histogram approximation,
# "slow" computes the statistic from sorted phases.
function kuiper(t::AbstractVector{T}, y::AbstractVector{T}, freqs::AbstractVector{T};
                dy::Union{AbstractVector{T},Nothing}=nothing, bins::Int=64, method::String="fast") where {T<:Real}
    N = length(t)
    y_avg = mean(y)
    wy = similar(y)
    
    if dy === nothing || isempty(dy)
        @inbounds for i in 1:N
            wy[i] = y[i] - y_avg
        end
        factor = 2 / sum(abs, wy)
        wy .*= factor
    else
        w = 1 ./(dy .^ 2)
        @inbounds for i in 1:N
            wy[i] = (y[i] - y_avg) * w[i]
        end
        @inbounds for i in 1:N
            wy[i] -= mean(wy)
        end
        factor = 2 / sum(abs, wy)
        wy .*= factor
    end

    Vn = zeros(eltype(y), length(freqs))
    
    if method == "slow"
        slow_phases   = similar(t)
        slow_sorted_wy = similar(wy)
        order         = Vector{Int}(undef, N)  # preallocated index vector for sortperm!
    end

    for (j, f) in enumerate(freqs)
        if method == "fast"
            binvals = zeros(eltype(y), bins)
            @inbounds for i in 1:N
                phase = t[i] * f - floor(t[i] * f)
                idx = clamp(floor(Int, bins * phase) + 1, 1, bins)
                binvals[idx] += wy[i]
            end
            cs = similar(binvals)
            cumsum!(cs, binvals)
        elseif method == "slow"
            @inbounds for i in 1:N
                slow_phases[i] = t[i] * f - floor(t[i] * f)
            end
            sortperm!(order, slow_phases)
            @inbounds for k in 1:N
                slow_sorted_wy[k] = wy[order[k]]
            end
            cs = similar(slow_sorted_wy)
            cumsum!(cs, slow_sorted_wy)
        else
            error("Unknown method argument. Choose 'fast' or 'slow'")
        end
        Vn[j] = maximum(cs) - minimum(cs)
    end

    return Vn
end
