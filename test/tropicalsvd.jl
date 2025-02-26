using Test
using TropicalNumbers
using Random
using TropicalSweepContractor
using TropicalSweepContractor: tropical_svd,bisec_svd

@testset "tropical_svd" begin
    m = 5
    n = 5
    Random.seed!(1234)
    c = [TropicalAndOr(rand()>0.5) for i in 1:m, j in 1:n]
    ans,a,b = tropical_svd(c,4)
    @test ans == true
    @test c == a*b
end

@testset "tropical_svd" begin
    m = 20
    k= 4
    n = 10
    Random.seed!(1234)
    a = [TropicalAndOr(rand()>0.5) for i in 1:m, j in 1:k]
    b = [TropicalAndOr(rand()>0.5) for i in 1:k, j in 1:n]
    c = a*b
    ans,a,b = tropical_svd(c,k)
    @test ans == true
    @test c == a*b
end


@testset "bisec_svd" begin
    m = 20
    k= 4
    n = 10
    Random.seed!(1234)
    a = [TropicalAndOr(rand()>0.5) for i in 1:m, j in 1:k]
    b = [TropicalAndOr(rand()>0.5) for i in 1:k, j in 1:n]
    c = a*b
    a,b,k = bisec_svd(c)
    @test c == a*b
    @test k == 4
end

@testset "bisec_svd" begin
    m = 10
    n = 10
    Random.seed!(1234)
    c = [TropicalAndOr(rand()>0.5) for i in 1:m, j in 1:n]
    k,a,b = bisec_svd(c)
    @test c == a*b
    @test k == 8
end