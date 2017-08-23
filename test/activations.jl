@testset "Sigmoid" begin
    xs = Float64[-1, 0, 1]
    @test sigmoid(xs) == [0.2689414213699951, 0.5, 0.7310585786300049]

    xs = Float32[-1, 0, 1]
    @test sigmoid(xs) == [0.26894143f0, 0.5f0, 0.7310586f0]

    # Ensure that σ works as well
    @test σ(Float64[0]) == [0.5]
end

@testset "relu" begin
    out = [0, 0, 0]
    @test relu!(out, [-1, 0, 1]) == [0, 0, 1]
    @test out == [0, 0, 1]

    @test relu([-1, 0, 1]) == [0, 0, 1]
end

@testset "elu" begin
    @test elu(Float64[-1, 0, 1]) == [exp(-1) - 1, 0, 1]
    @test elu(Float64[-1, 0, 1], .5) == [.5*(exp(-1) - 1), 0, 1]
end

@testset "selu" begin
    @test selu(Float64[-1, 0, 1]) == [-1.111327540011132, 0, 1.0507]
end

@testset "softmax" begin
    @test softmax(Float32[-1, 0, 1]) == [0.09003058f0, 0.24472848f0, 0.66524094f0]
end