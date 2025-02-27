# and_tensor(T::Type) = [[one(T) one(T); one(T) zero(T)];;; [zero(T) zero(T); zero(T) one(T)]]
# xor_tensor(T::Type) = [[one(T) zero(T); zero(T) one(T)];;; [zero(T) one(T); one(T) zero(T)]]
# or_tensor(T::Type) = [[one(T) zero(T); zero(T) zero(T)];;; [zero(T) one(T); one(T) one(T)]]
true_tensor(T::Type) = [zero(T); one(T)]
false_tensor(T::Type) = [one(T); zero(T)]
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

function factoring_tensornetwork(p_num,q_num,N;T::Type = Float64)
    edge_pi = collect(1:p_num)
    edge_si = collect(p_num+1:2*p_num)
    edge_qi = collect(2*p_num+1:2*p_num+q_num)
    edge_ci = collect(2*p_num+q_num+1:2*p_num+2*q_num)

    ptn = Vector{PlanarTensor{Int,T}}()
    edge_count = 2*p_num+2*q_num
    m_vec = Vector{Int}()
    x_vec = Vector{Float64}()
    y_vec = Vector{Float64}()

    for i in edge_pi
        push!(ptn,PlanarTensor(one_tensor(T),[i],Float64(i),0.0))
    end

    for i in edge_qi
        push!(ptn,PlanarTensor(one_tensor(T),[i],0.0,Float64(i)-2*p_num))
    end

    for i in edge_si
        push!(ptn,PlanarTensor(false_tensor(T),[i],i+0.5-p_num,0.0))
    end

    for i in edge_ci
        push!(ptn,PlanarTensor(false_tensor(T),[i],0.0,i+0.5-2*p_num-q_num))
    end

    for j in 1:q_num, i in 1:p_num
        push!(ptn,PlanarTensor(multiplier_tensor(T),[edge_pi[i], edge_count+1 , edge_qi[j],edge_count+2, edge_ci[j], edge_count+3, edge_si[i], edge_count+4],Float64(i),Float64(j)))
        edge_pi[i] = edge_count+1
        edge_qi[j] = edge_count+2
        if (i > 1)
            edge_si[i-1] = edge_count+4
        else
            push!(m_vec, edge_count+4)
            push!(x_vec, 0.0)
            push!(y_vec, j+0.75)
        end
        
        if i < q_num
            edge_ci[j] = edge_count+3
        else
            edge_si[i] = edge_count+3
        end
        edge_count += 4
    end

    for i in 1:p_num
        push!(ptn,PlanarTensor(one_tensor(T),[edge_pi[i]],Float64(i),q_num+1.0))
        push!(m_vec,edge_si[i])
        push!(x_vec,i+0.5)
        push!(y_vec,q_num+1.0)
    end

    for j in 1:q_num
        push!(ptn,PlanarTensor(one_tensor(T),[edge_qi[j]],p_num+1.0,Float64(j)))
    end

    for i in 1:length(m_vec)
        push!(ptn,PlanarTensor((N >> (i-1) & 1 == 1) ? true_tensor(T) : false_tensor(T),[m_vec[i]],x_vec[i],y_vec[i]))
    end
    return PlanarTensorNetwork(ptn,edge_count)
end
