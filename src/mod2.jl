function SweepContract.truncMPS!(mps::MPS{Mod{N,T}}, χ::Int64) where {N,T}
    L = length(mps)
    for i in 1:(L-1)
        p, l, u, q = lu_with_complete_pivot(reshape(mps[i], size(mps[i], 1) * size(mps[i], 2), :))
        mps[i] = reshape(p' * l, size(mps[i], 1), size(mps[i], 2), size(l, 2))
        mps[i+1] = reshape(u * q'*reshape(mps[i+1], (size(mps[i+1],1),
        size(mps[i+1],2)*size(mps[i+1],3))), (size(l, 2),size(mps[i+1],2),size(mps[i+1],3)))
    end

    for i in L:-1:2
        p, l, u, q = lu_with_complete_pivot(reshape(mps[i], :, size(mps[i], 2) * size(mps[i], 3)))
        mps[i] = reshape(u * q', size(u, 1), size(mps[i], 2), size(mps[i], 3))
        mps[i-1] = reshape(reshape(mps[i-1], (size(mps[i-1],1)*size(mps[i-1],2),size(mps[i-1],3))) * p'* l , (size(mps[i-1],1),size(mps[i-1],2), size(u, 1)))
    end
end

function lu_with_complete_pivot(A::AbstractMatrix{T}) where T
    n, m = size(A)

    P = Matrix{T}(I(n))
    Q = Matrix{T}(I(m))
    L = zeros(T, n, n)
    U = copy(A)


    ranka = min(n, m)
    for k in 1:min(n, m)
        # Find the pivot
        # index = argmax(abs.(U[k:end, k:end]))
        subU = view(U, k:n, k:m)
        # TODO: fix the bug in Mods
        index = findfirst(!iszero, subU)
        if isnothing(index)
            ranka = k - 1
            break
        end

        pivot_row, pivot_col = index[1], index[2]
        pivot_row += k - 1
        pivot_col += k - 1
        # Swap rows in U and P

        U[[k, pivot_row], :] = U[[pivot_row, k], :]
        L[[k, pivot_row], :] = L[[pivot_row, k], :]
        P[[k, pivot_row], :] = P[[pivot_row, k], :]

        # Swap columns in U and Q
        U[:, [k, pivot_col]] = U[:, [pivot_col, k]]
        Q[:, [k, pivot_col]] = Q[:, [pivot_col, k]]

        # Compute multipliers and update U
        for i in k+1:n
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:end] -= L[i, k] * U[k, k:end]
        end
    end

    # L[1:ranka, 1:ranka] += I(ranka)  # Add identity to L

    L += I(n) 
    return P, L[:, 1:ranka], U[1:ranka, :], Q
end