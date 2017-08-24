@testset "Affine" begin
    x = randn(10)
    W = randn(6, 10)
    b = randn(1)
    @test affine(x, W, b) == W * x .+ b
    @test infer_shape(affine, x, W, b) == (6,)
end