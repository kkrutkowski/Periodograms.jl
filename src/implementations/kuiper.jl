function kuiper(t::AbstractVector{T}, y::AbstractVector{T}, freqs::AbstractVector{T};
                      dy::Union{AbstractVector{T},Nothing}=nothing, bins::Int=64) where {T<:Real}
    N = length(t)
    # Compute mean of observations
    y_avg = mean(y)
    # Allocate array for the modified (normalized/weighted) values.
    wy = similar(y)

    if dy === nothing || isempty(dy)
        # Simple linear normalization: subtract the mean.
        for i in 1:N
            wy[i] = y[i] - y_avg
        end
        # Compute the absolute average deviation.
        AAD = sum(abs, wy)
        factor = 2 / AAD
        wy .*= factor
    else
        # Weighted linear normalization: weight differences by inverse variance.
        w = 1 ./(dy .^ 2)
        for i in 1:N
            wy[i] = (y[i] - y_avg) * w[i]
        end
        # Remove any overall bias by subtracting the weighted mean.
        wsum = mean(wy)
        for i in 1:N
            wy[i] -= wsum
        end
        AAD = sum(abs, wy)
        factor = 2 / AAD
        wy .*= factor
    end

    # Compute the approximate Kuiper statistic Vn for each frequency.
    Vn = zeros(eltype(y), length(freqs))
    for (j, f) in enumerate(freqs)
        binvals = zeros(eltype(y), bins)
        for i in 1:N
            # Compute the phase: fractional part of t[i] * f.
            phase = t[i] * f - floor(t[i] * f)
            # Convert phase to a bin index in 1:bins.
            idx = clamp(floor(Int, bins * phase) + 1, 1, bins)
            binvals[idx] += wy[i]
        end
        # Cumulative sum of bin values.
        cs = cumsum(binvals)
        Vn[j] = maximum(cs) - minimum(cs)
    end

    return Vn
end

