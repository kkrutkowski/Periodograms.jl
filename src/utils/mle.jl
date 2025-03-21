# Cholesky factorization: computes lower-triangular L such that A = L*L'
function llt(A)
    n = size(A, 1)
    L = zeros(eltype(A), n, n)
    for i in 1:n
        @inbounds for j in 1:i
            s = 0.0
            for k in 1:j-1
                s += L[i, k] * L[j, k]
            end
            if i == j
                x = A[i, i] - s
                if x > 0
                    L[i, j] = sqrt(A[i, i] - s)
                else
                    L[i, j] = 0
                end
            else
                L[i, j] = (A[i, j] - s) / L[j, j]
            end
        end
    end
    return L
end

# Forward substitution: solve L*y = b for y using explicit loops.
function forward_substitution(L, b)
    n = length(b)
    y = zeros(eltype(b), n)
    @inbounds for i in 1:n
        s = 0.0
        @inbounds for j in 1:i-1
            s += L[i, j] * y[j]
        end
        y[i] = (b[i] - s) / L[i, i]
    end
    return y
end

# Backward substitution: solve L'*x = y for x using explicit loops.
function backward_substitution(L, y)
    n = length(y)
    x = zeros(eltype(y), n)
    @inbounds for i in n:-1:1
        s = 0.0
        @inbounds for j in i+1:n
            # Since (L')_{i,j} = L[j, i]
            s += L[j, i] * x[j]
        end
        x[i] = (y[i] - s) / L[i, i]
    end
    return x
end

# Cholesky solver: solve Ax = b using the manually computed factor L,
# with explicit nested loops for forward and backward substitutions.
function solve(A, b)
    L = llt(A)
    y = forward_substitution(L, b)
    x = backward_substitution(L, y)
    return x
end
