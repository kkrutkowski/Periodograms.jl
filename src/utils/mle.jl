# Define a type alias for an 8-element vector of Float64
const Vec8 = Vector{Float64}

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
    LDL = [zeros(8) for _ in 1:n, _ in 1:n]

    @inbounds for i in 1:n
        # Calculate the off-diagonal elements of L for row i
        for j in 1:i-1
            s = zeros(8)
            for k in 1:j-1
                # Element-wise multiplication
                for sys in 1:8
                    s[sys] += LDL[i, k][sys] * LDL[j, k][sys] * LDL[k, k][sys]
                end
            end
            # Element-wise division
            for sys in 1:8
                LDL[i, j][sys] = (A[i, j][sys] - s[sys]) / LDL[j, j][sys]
            end
        end

        # Calculate the i-th diagonal element of D
        s_diag = zeros(8)
        for k in 1:i-1
            for sys in 1:8
                s_diag[sys] += LDL[i, k][sys] * LDL[i, k][sys] * LDL[k, k][sys]
            end
        end
        # Element-wise subtraction
        for sys in 1:8
            LDL[i, i][sys] = A[i, i][sys] - s_diag[sys]
        end
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
    y = [zeros(8) for _ in 1:n]

    @inbounds for i in 1:n
        s = zeros(8)
        for j in 1:i-1
            for sys in 1:8
                s[sys] += LDL[i, j][sys] * y[j][sys]
            end
        end
        for sys in 1:8
            y[i][sys] = b[i][sys] - s[sys]
        end
    end

    return y
end


"""
    diagonal_solve(LDL, y)

Solves D*z = y for z, where D is the diagonal matrix from the LDLT decomposition.
"""
function diagonal_solve(LDL::Matrix{Vec8}, y::Vector{Vec8})
    n = length(y)
    z = [zeros(8) for _ in 1:n]

    @inbounds for i in 1:n
        for sys in 1:8
            z[i][sys] = y[i][sys] / LDL[i, i][sys]
        end
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
    x = [zeros(8) for _ in 1:n]

    @inbounds for i in n:-1:1
        s = zeros(8)
        for j in i+1:n
            for sys in 1:8
                s[sys] += LDL[j, i][sys] * x[j][sys]
            end
        end
        for sys in 1:8
            x[i][sys] = z[i][sys] - s[sys]
        end
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
function solve(A::Matrix{Vec8}, b::Vector{Vec8})
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

# Define a type alias for an 8-element vector of Float64
const Vec8 = Vector{Float64}

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
    LDL = [zeros(8) for _ in 1:n, _ in 1:n]

    @inbounds for i in 1:n
        # Calculate the off-diagonal elements of L for row i
        for j in 1:i-1
            s = zeros(8)
            for k in 1:j-1
                # Element-wise multiplication
                for sys in 1:8
                    s[sys] += LDL[i, k][sys] * LDL[j, k][sys] * LDL[k, k][sys]
                end
            end
            # Element-wise division
            for sys in 1:8
                LDL[i, j][sys] = (A[i, j][sys] - s[sys]) / LDL[j, j][sys]
            end
        end

        # Calculate the i-th diagonal element of D
        s_diag = zeros(8)
        for k in 1:i-1
            for sys in 1:8
                s_diag[sys] += LDL[i, k][sys] * LDL[i, k][sys] * LDL[k, k][sys]
            end
        end
        # Element-wise subtraction
        for sys in 1:8
            LDL[i, i][sys] = A[i, i][sys] - s_diag[sys]
        end
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
    y = [zeros(8) for _ in 1:n]

    @inbounds for i in 1:n
        s = zeros(8)
        for j in 1:i-1
            for sys in 1:8
                s[sys] += LDL[i, j][sys] * y[j][sys]
            end
        end
        for sys in 1:8
            y[i][sys] = b[i][sys] - s[sys]
        end
    end

    return y
end


"""
    diagonal_solve(LDL, y)

Solves D*z = y for z, where D is the diagonal matrix from the LDLT decomposition.
"""
function diagonal_solve(LDL::Matrix{Vec8}, y::Vector{Vec8})
    n = length(y)
    z = [zeros(8) for _ in 1:n]

    @inbounds for i in 1:n
        for sys in 1:8
            z[i][sys] = y[i][sys] / LDL[i, i][sys]
        end
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
    x = [zeros(8) for _ in 1:n]

    @inbounds for i in n:-1:1
        s = zeros(8)
        for j in i+1:n
            for sys in 1:8
                s[sys] += LDL[j, i][sys] * x[j][sys]
            end
        end
        for sys in 1:8
            x[i][sys] = z[i][sys] - s[sys]
        end
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
function solve(A::Matrix{Vec8}, b::Vector{Vec8})
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

# Define a type alias for an 8-element vector of ComplexF64
const CVec8 = Vector{ComplexF64}

"""
    solve_levinson!(x, R, y, a, a_prev)

Vectorized in-place O(N^2) Levinson-Durbin recursion for solving 8 systems Tx = y
simultaneously, where T is a Hermitian Toeplitz matrix defined by its first column R.

# Arguments
- `x::Vector{CVec8}`: Solution vector (pre-allocated).
- `R::Vector{CVec8}`: First column of the Toeplitz matrices.
- `y::Vector{CVec8}`: Right-hand side vectors.
- `a::Vector{CVec8}`: Predictor coefficients (pre-allocated).
- `a_prev::Vector{CVec8}`: Previous predictor coefficients (pre-allocated).
"""
function solve_levinson!(x::Vector{CVec8}, R::Vector{CVec8}, y::Vector{CVec8},
                         a::Vector{CVec8}, a_prev::Vector{CVec8})
    n = length(R)

    # Initialize k = 1
    # E is real, so we use a standard Float64 vector of size 8
    E = zeros(Float64, 8)

    @inbounds for sys in 1:8
        E[sys] = real(R[1][sys])
        x[1][sys] = y[1][sys] / R[1][sys]
        a[1][sys] = 1.0 + 0.0im
    end

    @inbounds for k in 1:n-1
        # Cross-correlation for predictor
        lambda = zeros(ComplexF64, 8)
        for i in 1:k
            for sys in 1:8
                lambda[sys] += R[k + 2 - i][sys] * a[i][sys]
            end
        end

        gamma = zeros(ComplexF64, 8)
        for sys in 1:8
            gamma[sys] = -lambda[sys] / E[sys]
        end

        # Update predictor coefficients (a)
        for i in 1:k
            for sys in 1:8
                a_prev[i][sys] = a[i][sys]
            end
        end

        for sys in 1:8
            a[k+1][sys] = gamma[sys] * conj(a_prev[1][sys])
        end

        for i in 2:k
            for sys in 1:8
                a[i][sys] = a_prev[i][sys] + gamma[sys] * conj(a_prev[k - i + 2][sys])
            end
        end

        for sys in 1:8
            E[sys] *= (1.0 - abs2(gamma[sys]))
        end

        # Discrepancy for solution
        mu = zeros(ComplexF64, 8)
        for sys in 1:8
            mu[sys] = y[k + 1][sys]
        end

        for i in 1:k
            for sys in 1:8
                mu[sys] -= R[k + 2 - i][sys] * x[i][sys]
            end
        end

        nu = zeros(ComplexF64, 8)
        for sys in 1:8
            nu[sys] = mu[sys] / E[sys]
        end

        # Update solution vector (x)
        for sys in 1:8
            x[k+1][sys] = nu[sys] * conj(a[1][sys])
        end

        for i in 1:k
            for sys in 1:8
                x[i][sys] += nu[sys] * conj(a[k + 2 - i][sys])
            end
        end
    end
end
