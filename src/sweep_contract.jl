const MPSTensor{T} = Array{T,3}
const MPS{T} = Vector{MPSTensor{T}}

function truncMPS!(M::MPS{T}, χ::Int64) where T <: Real
    # Put the MPS in canonical form using the QR decomposition, sweeping left-to-right
    for i ∈ 1:length(M)-1
        X = reshape(M[i],(size(M[i],1)*size(M[i],2),size(M[i],3)))
        q,r = LinearAlgebra.qr(X)
        if size(r,1)==size(r,2)
            M[i] = reshape(Matrix(q),size(M[i]))
            LinearAlgebra.lmul!(LinearAlgebra.UpperTriangular(r),
                reshape(M[i+1],(size(M[i+1],1),size(M[i+1],2)*size(M[i+1],3))))
        else
            M[i] = reshape(Matrix(q),(size(M[i],1),size(M[i],2),size(r,1)))
            M[i+1] = reshape(r*reshape(M[i+1], (size(M[i+1],1),
                size(M[i+1],2)*size(M[i+1],3))), (size(r,1),size(M[i+1],2),size(M[i+1],3)))
        end
    end
    # Perform the bond truncation using the SVD decomposition, sweeping right-to-left
    for i ∈ length(M):-1:2
        X = reshape(M[i],(size(M[i],1),size(M[i],2)*size(M[i],3)))
        # In some rare cases the default svd can fail to converge
        try
            F = LinearAlgebra.svd!(X);
            (u,s,v) = (F.U,F.S,F.V)
        catch _
            F = LinearAlgebra.svd!(X; alg=LinearAlgebra.QRIteration())
            (u,s,v) = (F.U,F.S,F.V)
        end
        b = min(length(s),χ)
        u = u[:,1:b]
        s = s[1:b]
        v = v[:,1:b]'
        M[i] = reshape(v,(b,size(M[i],2),size(M[i],3)));
        X = reshape(M[i-1],(size(M[i-1],1)*size(M[i-1],2),size(M[i-1],3)))*u
        LinearAlgebra.rmul!(X,LinearAlgebra.Diagonal(s));
        M[i-1] = reshape(X,(size(M[i-1],1),size(M[i-1],2),b));
    end
    return M
end

function sweep_contract!(ptn::PlanarTensorNetwork{T}, χ::Int, τ::Int) where T
    ptn = sort_ptn(ptn)

    l2t = label2tensor(ptn)

    resexp = 0
    count = 0

    local MPS_t
    local mps_edges2mps
    
    mps_edges = Int[]

    for (i,t) ∈ enumerate(ptn.tensors)
        ind_up = Int[]
        ind_do = Int[]
        for n ∈ t.labels
            if n in mps_edges
                push!(ind_do, n)
            else
                push!(ind_up, n)
            end
        end
        sort!(ind_up, by=λ->atan(ptn[setdiff(l2t[λ],i)[1]].x-t.x,ptn[setdiff(l2t[λ],i)[1]].y-t.y))
        sort!(ind_do, by=λ->mps_edges2mps[λ])
        σ = permutebetween(t.labels, [ind_do; ind_up])
        tensor = permutedims(t.tensor, σ)

        if isempty(mps_edges)
            MPS_t = splitMPStensor(reshape(tensor,(1,size(t.tensor)...,1)))
            mps_edges = ind_up
            mps_edges2mps = Dict{Int,Int}(n=>i for (i,n) ∈ enumerate(ind_up))
        else
            mps_min = mps_edges2mps[ind_do[1]]
            mps_max = mps_edges2mps[ind_do[end]]
            @assert mps_max-mps_min + 1 == length(ind_do)
            code = get_eincode(mps_edges[mps_min:mps_max], [ind_do; ind_up],ind_up)
            contract_tensor = einsum(code,(MPS_t[mps_min:mps_max]..., tensor))
            if isempty(ind_up)
                if mps_min > 1
                    code = EinCode([[-1,mps_edges[mps_min-1],-2],[-2,-3]],[-1,mps_edges[mps_min-1],-3])
                    contract_tensor = einsum(code,(MPS_t[mps_min-1],contract_tensor))
                    MPS_t = [MPS_t[1:mps_min-2]; [contract_tensor]; MPS_t[mps_max+1:end]]
                    mps_edges = [mps_edges[1:mps_min-2]; mps_edges[mps_min-1]; mps_edges[mps_max+1:end]]
                elseif mps_max < length(MPS_t)
                    code = EinCode([[-2,mps_edges[mps_max+1],-1],[-3,-2]],[-3,mps_edges[mps_max+1],-1])
                    contract_tensor = einsum(code,(MPS_t[mps_max+1],contract_tensor))
                    MPS_t = [MPS_t[1:mps_min-1]; [contract_tensor]; MPS_t[mps_max+2:end]]
                    mps_edges = [mps_edges[1:mps_min-1]; mps_edges[mps_max+1]; mps_edges[mps_max+2:end]]
                else
                    return contract_tensor[1][1]
                end
            else
                MPS_t = [MPS_t[1:mps_min-1]; splitMPStensor(contract_tensor); MPS_t[mps_max+1:end]]
                mps_edges = [mps_edges[1:mps_min-1]; ind_up; mps_edges[mps_max+1:end]]
            end
            mps_edges2mps = Dict{Int,Int}(n=>i for (i,n) ∈ enumerate(mps_edges))

            @show size.(MPS_t,3)
            if any(size.(MPS_t,3).>τ)
                truncMPS!(MPS_t, χ)
                @show "after trunc"
                @show size.(MPS_t,3)
            end
        end
    end
    return MPS_t
    report && println("Number of truncations: $count")

    res = MPS_t[1][1];
    if res == 0.0
        return (0.0, typemin(Int64));
    end
    h = Int(floor(log2(abs(res))));
    return (res/exp2(h),resexp+h);
end

function splitMPStensor(tensor::Array{T}) where T
    v = MPS{T}(undef,ndims(tensor)-2)
    (l,r) = (2,ndims(tensor)-1)
    s = collect(size(tensor))
    while r>l
        # Calculate the bond dimensions of sweeping tensor either way and minimise
        L = s[l-1]*s[l]
        R = s[r]*s[r+1]
        if L <= R
            v[l-1] = MPSTensor{T}(reshape(diagm(fill(one(T),L)),(s[l-1],s[l],L)))
            s[l] *= s[l-1]
            l += 1
        else
            # v[r-1] = reshape(Matrix{T}(LinearAlgebra.I,R,R),(R,s[r],s[r+1]));
            v[r-1] = MPSTensor{T}(reshape(diagm(fill(one(T),R)),(R,s[r],s[r+1])))
            s[r] *= s[r+1]
            r -= 1
        end
    end
    v[l-1] = MPSTensor{T}(reshape(tensor,(s[l-1],s[l],s[l+1])))
    return v
end

function permutebetween(from, to)
    σf = sortperm(from)
    σt = sortperm(to)
    arr = Vector{Int}(undef,length(from))
    for i ∈ eachindex(from)
        arr[σt[i]]=σf[i]
    end
    return arr
end

function get_eincode(mps_vec, tensor_ix,ind_up)
    mps_ix = [[-i,t,-i-1] for (i,t) in enumerate(mps_vec)]
    return EinCode([mps_ix..., tensor_ix], [-1,ind_up...,-length(mps_vec)-1])
end

function truncMPS!(M::MPS{TropicalAndOr}, χ::Int64)
    for i in 1:length(M)-1
        X = reshape(M[i],(size(M[i],1)*size(M[i],2),size(M[i],3)))
        k,a,b = bisec_svd(X)
        M[i] = reshape(a,(size(M[i],1),size(M[i],2),k))
        M[i+1] = reshape(b*reshape(M[i+1], (size(M[i+1],1),
            size(M[i+1],2)*size(M[i+1],3))), (k,size(M[i+1],2),size(M[i+1],3)))
    end
end