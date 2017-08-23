@testset "Affine" begin
    x = randn(10)
    W = randn(10, 10)
    b = randn(10)
    @test affine(x, W, b) == W * x + b
end