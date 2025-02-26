function tropical_svd(C::Matrix{TropicalAndOr},k::Int;verbose = false)
    # C = A * B
    # A: m x k, B: k x n
    # IP by JuMP

    m,n = size(C)
    model = Model(HiGHS.Optimizer)
    !verbose && set_silent(model)

    @variable(model, 0 <= a[i = 1:m*k] <= 1, Int) # a[i,l] = a[(l-1)*m+i]
    @variable(model, 0 <= b[i = 1:k*n] <= 1, Int)   # b[l,j] = b[(j-1)*k+l]
    @variable(model, 0 <= d[i = 1:m*k*n] <= 1, Int) # d[i,l,j] = d[(j-1)*k*m+(l-1)*m+i]
    @objective(model, Min,1)
    
    for i in 1:m
        for j in 1:n
            for l in 1:k
                @constraint(model, d[(j-1)*k*m+(l-1)*m+i] <= a[(l-1)*m+i])
                @constraint(model, d[(j-1)*k*m+(l-1)*m+i] <= b[(j-1)*k+l])
                @constraint(model, d[(j-1)*k*m+(l-1)*m+i] + 1 >= a[(l-1)*m+i] +b[(j-1)*k+l])
                @constraint(model, d[(j-1)*k*m+(l-1)*m+i] <= (C[i,j].n ? 1 : 0))
            end
            @constraint(model, sum(d[(j-1)*k*m+(l-1)*m+i] for l in 1:k) >= (C[i,j].n ? 1 : 0))
        end
    end
    
    optimize!(model)
    return  is_solved_and_feasible(model),reshape([TropicalAndOr(v ≈ 1.0) for v in value.(a)],m,k), reshape([TropicalAndOr(v ≈ 1.0) for v in value.(b)],k,n)
end
function bisec_svd(C::Matrix{TropicalAndOr})
    m,n = size(C)
    if m == 1 || n == 1
        return true, C, C
    end
    ans = false
    kmin = 1
    kmax = min(m,n)
    k = kmax
    local success_a,success_b
    while kmin < kmax
        k = (kmin + kmax) ÷ 2
        ans,a,b = tropical_svd(C,k)
        @assert !ans || (C == a*b)
        if ans
            kmax = k
            success_a = a
            success_b = b
        else
            kmin = k + 1
        end
    end
    if kmin == min(m,n)
        ans,success_a,success_b = tropical_svd(C,kmin)
    end
    return kmin,success_a,success_b
end