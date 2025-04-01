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

# Generate a time vector and observations
t = 0:0.01:100            # 0 to 100 seconds with a 0.01 sec sampling interval
f0 = 1.0                  # true frequency in Hz
y = sin.(2π * f0 .* t) .+ 0.5 * randn(length(t))  # Sinusoid plus Gaussian noise
freqs = 0.0:0.0005:2.5

# Convert t and freqs to vectors for consistency in benchmarking.
t_vec = collect(t)
freqs_vec = collect(freqs)

# Compute the Kuiper statistic for each frequency using both methods.
Vn_fast = kuiper(t_vec, y, freqs_vec, method="fast")
Vn_slow = kuiper(t_vec, y, freqs_vec, method="slow")

println("Results are approximately equal: ", isapprox(Vn_fast, Vn_slow, atol=1e-1))

# Plot both methods.
plt = plot(freqs_vec, Vn_fast, xlabel="Frequency (Hz)", ylabel="Kuiper Statistic Vₙ",
           title="Kuiper Test Statistic vs Frequency", label="Fast Method", ylim=(0.0,1.0))
plot!(plt, freqs_vec, Vn_slow, label="Slow Method")
display(plt)

# Benchmark the two methods.
println("\nBenchmarking the Fast Method:")
@btime kuiper($t_vec, $y, $freqs_vec, method="fast");

println("\nBenchmarking the Slow Method:")
@btime kuiper($t_vec, $y, $freqs_vec, method="slow");

println("Close the plot window or press Enter to exit.")
readline()
using Statistics
using Plots
using Random
using BenchmarkTools

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
