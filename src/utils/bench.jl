using BenchmarkTools
using Random

include("trig_sum.jl")

# Ensure reproducibility
Random.seed!(1234)

# Number of measurements
N_measurements = 1024

# Generate synthetic measurement data:
# Sorted times and corresponding random weights
t = sort(rand(N_measurements))
h = rand(N_measurements)

# Frequency parameters
df = 1.0    # frequency spacing
f0 = 0.0    # initial frequency

# List of grid lengths: 1024, 2048, 4096, …, 1024*128 = 131072
grid_sizes = [1024 * 2^(i) for i in 0:7]

println("\nBenchmarking FFT-free branch (use_fft = false):")
for N in grid_sizes
    println("Grid size: ", N)
    # Benchmark using FFT-free (direct computation) branch
    @btime trig_sum($t, $h, $df, $N; f0=$f0, use_fft=false);
end


println("Benchmarking FFT branch (use_fft = true):")
for N in grid_sizes
    println("Grid size: ", N)
    # Benchmark using FFT branch
    @btime trig_sum($t, $h, $df, $N; f0=$f0, use_fft=true);
end
