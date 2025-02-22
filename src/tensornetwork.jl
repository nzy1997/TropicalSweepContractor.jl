or_tensor(T::Type) = [[one(T) zero(T); zero(T) zero(T)];;; [zero(T) one(T); one(T) one(T)]]
true_tensor(T::Type) = [zero(T); one(T)]
false_tensor(T::Type) = [one(T); zero(T)]
and_tensor(T::Type) = [[one(T) one(T); one(T) zero(T)];;; [zero(T) zero(T); zero(T) one(T)]]
xor_tensor(T::Type) = [[one(T) zero(T); zero(T) one(T)];;; [zero(T) one(T); one(T) zero(T)]]
one_tensor(T::Type) = [one(T) ;one(T)]

function multiplier_tensor(T::Type)
    mat = zeros(T, fill(2, 8)...)
    for num = 0:2^8-1
        p_i = num & 1
        p_o = (num >> 1) & 1
        q_i = (num >> 2) & 1
        q_o = (num >> 3) & 1
        c_i = (num >> 4) & 1
        c_o = (num >> 5) & 1
        s_i = (num >> 6) & 1
        s_o = (num >> 7) & 1
        if (p_i == p_o) && (q_i == q_o) && (2 * c_o + s_o == p_i * q_i + c_i + s_i)
           mat[num+1] = one(T)
        end
    end
    return mat
end

struct PlanarTensorNetwork{T<:Integer,T2}
    pmg::PlanarMultigraph{T}
    tensors::Dict{T,Array{T2}} # v_id -> tensor
end

function factoring_tensornetwork(p_num,q_num,N;T::Type = Float64)
    m_num = q_num + p_num

    edge_pi = collect(1:p_num)
    edge_si = collect(p_num+1:2*p_num)
    edge_qi = collect(2*p_num+1:2*p_num+q_num)
    edge_ci = collect(2*p_num+q_num+1:2*p_num+2*q_num)

    vec_tensor = Dict{Int,Array{T}}()
    vec_lable = Dict{Int,Vector{Int}}()

    tensor_count = 0
    edge_count = 2*p_num+2*q_num
    m_vec = Vector{Int}()

    for i in edge_pi ∪ edge_qi
        tensor_count += 1
        vec_tensor[tensor_count] = one_tensor(T)
        vec_lable[tensor_count] = [i]
    end

    for i in edge_si ∪ edge_ci
        tensor_count += 1
        vec_tensor[tensor_count] = false_tensor(T)
        vec_lable[tensor_count] = [i]
    end

    for j in 1:q_num, i in 1:p_num
        tensor_count += 1
        vec_tensor[tensor_count] = multiplier_tensor(T)
        vec_lable[tensor_count] = [edge_pi[i], edge_count+1 , edge_qi[j],edge_count+2, edge_ci[j], edge_count+3, edge_si[j], edge_count+4]
        edge_pi[i] = edge_count+1
        edge_qi[j] = edge_count+2
        if (i > 1)
            edge_si[i-1] = edge_count+4
        else
            push!(m_vec, edge_count+4)
        end
        
        if i < q_num
            edge_ci[j] = edge_count+3
        else
            edge_si[i] = edge_count+3
        end
        edge_count += 4
    end

    for i in 1:p_num
        tensor_count += 1
        vec_tensor[tensor_count] = one_tensor(T)
        vec_lable[tensor_count] = [edge_pi[i]]
        push!(m_vec,edge_si[i])
    end

    for j in 1:q_num
        tensor_count += 1
        vec_tensor[tensor_count] = one_tensor(T)
        vec_lable[tensor_count] = [edge_qi[j]]
    end

    for i in 1:length(m_vec)
        tensor_count += 1
        vec_tensor[tensor_count] = (N >> (i-1) & 1 == 1) ? true_tensor(T) : false_tensor(T)
        vec_lable[tensor_count] = [m_vec[i]]
    end
    @show edge_count
    @show tensor_count
    @show m_vec
end