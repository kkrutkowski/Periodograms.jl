using StaticArrays

# Define a type alias for an 8-element vector of Float64 for clarity.
# Operations on SVector are highly optimized and will be vectorized by the compiler.
const Vec8 = SVector{8, Float64}

"""
    ldlt_decomposition(A)

Performs a vectorized LDLT decomposition for 8 systems simultaneously.
This computes the factorization A = L*D*L', where L is a unit lower
triangular matrix and D is a diagonal matrix.

# Arguments
- `A::Matrix{Vec8}`: A matrix of 8-element vectors representing 8 SPD matrices.

# Returns
- `LDL::Matrix{Vec8}`: A matrix where the strict lower triangle contains the
  elements of L and the diagonal contains the elements of D.
"""
function ldlt_decomposition(A::Matrix{Vec8})
    n = size(A, 1)
    LDL = zeros(Vec8, n, n)

    @inbounds for i in 1:n
        # Calculate the off-diagonal elements of L for row i
        for j in 1:i-1
            s = zero(Vec8)
            for k in 1:j-1
                # s += L[i, k] * D[k, k] * L[j, k]
                s += LDL[i, k] * LDL[j, k] * LDL[k, k]
            end
            # L[i, j] = (A[i, j] - s) / D[j, j]
            LDL[i, j] = (A[i, j] - s) / LDL[j, j]
        end

        # Calculate the i-th diagonal element of D
        s_diag = zero(Vec8)
        for k in 1:i-1
            # s_diag += L[i, k]^2 * D[k, k]
            s_diag += LDL[i, k] * LDL[i, k] * LDL[k, k]
        end
        # D[i, i] = A[i, i] - s_diag
        LDL[i, i] = A[i, i] - s_diag
    end

    return LDL
end

"""
    forward_substitution(LDL, b)

Solves L*y = b for y, where L is the unit lower triangular matrix from the LDLT
decomposition.
"""
function forward_substitution(LDL::Matrix{Vec8}, b::Vector{Vec8})
    n = length(b)
    y = zeros(Vec8, n)

    @inbounds for i in 1:n
        s = zero(Vec8)
        for j in 1:i-1
            s += LDL[i, j] * y[j]
        end
        y[i] = b[i] - s
    end

    return y
end

"""
    diagonal_solve(LDL, y)

Solves D*z = y for z, where D is the diagonal matrix from the LDLT decomposition.
"""
function diagonal_solve(LDL::Matrix{Vec8}, y::Vector{Vec8})
    n = length(y)
    z = zeros(Vec8, n)

    @inbounds for i in 1:n
        z[i] = y[i] / LDL[i, i]
    end

    return z
end

"""
    backward_substitution(LDL, z)

Solves L'*x = z for x, where L' is the transpose of the unit lower triangular matrix
from the LDLT decomposition.
"""
function backward_substitution(LDL::Matrix{Vec8}, z::Vector{Vec8})
    n = length(z)
    x = zeros(Vec8, n)

    @inbounds for i in n:-1:1
        s = zero(Vec8)
        for j in i+1:n
            # This uses L[j, i], which corresponds to (L')_{i,j}
            s += LDL[j, i] * x[j]
        end
        x[i] = z[i] - s
    end

    return x
end

"""
    solve(A, b)

Solves 8 systems of linear equations A*x = b simultaneously using a vectorized
LDLT decomposition. This function is the Julia equivalent of the C implementation
provided.

# Arguments
- `A::Matrix{SVector{8, Float64}}`: A square matrix where each element is an
  8-element vector, representing 8 symmetric positive-definite matrices.
- `b::Vector{SVector{8, Float64}}`: A vector where each element is an
  8-element vector, representing 8 right-hand side vectors.

# Returns
- `x::Vector{SVector{8, Float64}}`: The solution vector, where each element is
  an 8-element vector representing the 8 solutions.
"""
function solve(A::Matrix{SVector{8, Float64}}, b::Vector{SVector{8, Float64}})
    # Step 1: Perform LDLT decomposition: A = L*D*L'
    LDL = ldlt_decomposition(A)

    # Step 2: Solve L*y = b (forward substitution)
    y = forward_substitution(LDL, b)

    # Step 3: Solve D*z = y (diagonal scaling)
    z = diagonal_solve(LDL, y)

    # Step 4: Solve L'*x = z (backward substitution)
    x = backward_substitution(LDL, z)

    return x
end
