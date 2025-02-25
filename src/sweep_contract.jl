struct MPSTensor{T}
    tensor :: Array{T,3}
    out_edge :: Int
end
const MPS{T} = Vector{MPSTensor{T}}

function sweep_contract!(ptn::PlanarTensorNetwork{T}) where T
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
        # σ = permutebetween(t.adj, [ind_do; ind_up])
        # t.arr = permutedims(t.arr, σ)
        # s = size(t.arr)
        # t.arr = reshape(t.arr,(prod(s[1:length(ind_do)]),s[length(ind_do)+1:end]...))

        if isempty(mps_edges)
            MPS_t = splitMPStensor(reshape(tensor,(1,size(t.tensor)...,1)),ind_up)
            mps_edges = ind_up
            mps_edges2mps = Dict{Int,Int}(n=>i for (i,n) ∈ enumerate(ind_up))
        else
            mps_min = mps_edges2mps[ind_do[1]]
            mps_max = mps_edges2mps[ind_do[end]]
            @assert mps_max-mps_min + 1 == length(ind_do)
            code = get_eincode(getproperty.(MPS_t[mps_min:mps_max],:out_edge), [ind_do; ind_up],ind_up)
            contract_tensor = einsum(code,(getproperty.(MPS_t[mps_min:mps_max],:tensor)..., tensor))
            @show contract_tensor
            if isempty(ind_up)
                if mps_min > 1
                    code = EinCode([[-1,mps_edges[mps_min-1],-2],[-2,-3]],[-1,mps_edges[mps_min-1],-3])
                    contract_tensor = einsum(code,(MPS_t[mps_min-1].tensor,contract_tensor))
                    MPS_t = [MPS_t[1:mps_min-2]; MPSTensor(contract_tensor,mps_edges[mps_min-1]); MPS_t[mps_max+1:end]]
                    mps_edges = [mps_edges[1:mps_min-2]; mps_edges[mps_min-1]; mps_edges[mps_max+1:end]]
                elseif mps_max < length(MPS_t)
                    code = EinCode([[-2,mps_edges[mps_max+1],-1],[-3,-2]],[-3,mps_edges[mps_max+1],-1])
                    contract_tensor = einsum(code,(MPS_t[mps_max+1].tensor,contract_tensor))
                    MPS_t = [MPS_t[1:mps_min-1]; MPSTensor(contract_tensor,mps_edges[mps_max+1]); MPS_t[mps_max+2:end]]
                    mps_edges = [mps_edges[1:mps_min-1]; mps_edges[mps_max+1]; mps_edges[mps_max+2:end]]
                else
                    return contract_tensor[1][1]
                end
            else
                MPS_t = [MPS_t[1:mps_min-1]; splitMPStensor(contract_tensor,ind_up); MPS_t[mps_max+1:end]]
                mps_edges = [mps_edges[1:mps_min-1]; ind_up; mps_edges[mps_max+1:end]]
            end
            mps_edges2mps = Dict{Int,Int}(n=>i for (i,n) ∈ enumerate(mps_edges))
            # isnothing(lo) && throw(InvalidTNError("Disconnected TN"))

            # X::Array{Float64} = MPS_t[lo]
            # for j ∈ lo+1:hi
            #     finalsize = (size(X,1),size(X,2)*size(MPS_t[j],2),size(MPS_t[j],3))
            #     X = reshape(X,(size(X,1)*size(X,2),size(X,3)))*
            #         reshape(MPS_t[j],(size(MPS_t[j],1),size(MPS_t[j],2)*size(MPS_t[j],3)))
            #     X = reshape(X,finalsize)
            # end
            # X = permutedims(X,[1,3,2])
            # M = reshape(t.arr,(size(t.arr,1),prod(size(t.arr)[2:end])))
            # X = reshape(
            #     reshape(X,(size(X,1)*size(X,2),size(X,3)))*M,
            #     (size(X,1),size(X,2),size(M,2))
            # )
            # X = permutedims(X,[1,3,2])
            # X = reshape(X,(size(X,1),size(t.arr)[2:end]...,size(X,3)))

            # MPS_i = [MPS_i[1:lo-1]; ind_up; MPS_i[hi+1:end]]
            # if ndims(X)!=2
            #     MPS_t = [MPS_t[1:lo-1]; splitMPStensor(X); MPS_t[hi+1:end]]
            # elseif isempty(MPS_i)
            #     MPS_t=[reshape([X[1]],(1,1,1))]
            # elseif lo>1
            #     s = size(MPS_t[lo-1])
            #     MPS_t[lo-1] = reshape(
            #         reshape(MPS_t[lo-1],(s[1]*s[2],s[3]))*X,
            #         (s[1],s[2],size(X,2))
            #     )
            #     MPS_t = [MPS_t[1:lo-1]; MPS_t[hi+1:end]]
            # else
            #     s = size(MPS_t[hi+1])
            #     MPS_t[hi+1] = reshape(
            #         X*reshape(MPS_t[hi+1],(s[1],s[2]*s[3])),
            #         (size(X,1),s[2],s[3])
            #     )
            #     MPS_t = [MPS_t[1:lo-1]; MPS_t[hi+1:end]]
            # end

            # if any(size.(MPS_t,3).>τ)
            #     count += 1
            #     truncMPS!(MPS_t, χ)
            #     if LinearAlgebra.norm(MPS_t[1])==0
            #         return (0.0,typemin(Int))
            #     end
            #     h = Int(floor(log2(LinearAlgebra.norm(MPS_t[1]))))
            #     resexp += h
            #     MPS_t[1] /= exp2(h)
            # end
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

function splitMPStensor(tensor::Array{T},outer_indices::Vector{Int}) where T
    v = MPS{T}(undef,ndims(tensor)-2)
    (l,r) = (2,ndims(tensor)-1)
    s = collect(size(tensor))
    while r>l
        # Calculate the bond dimensions of sweeping tensor either way and minimise
        L = s[l-1]*s[l]
        R = s[r]*s[r+1]
        if L <= R
            v[l-1] = MPSTensor{T}(reshape(Matrix{T}(LinearAlgebra.I,L,L),(s[l-1],s[l],L)),outer_indices[l-1])
            s[l] *= s[l-1]
            l += 1
        else
            # v[r-1] = reshape(Matrix{T}(LinearAlgebra.I,R,R),(R,s[r],s[r+1]));
            v[r-1] = MPSTensor{T}(reshape(Matrix{T}(LinearAlgebra.I,R,R),(R,s[r],s[r+1])),outer_indices[r-1])
            s[r] *= s[r+1]
            r -= 1
        end
    end
    v[l-1] = MPSTensor{T}(reshape(tensor,(s[l-1],s[l],s[l+1])),outer_indices[l-1])
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