@testset "shape_inference" begin
    x = randn(100, 20)
    W = randn(4, 100)
    b = randn(1)
    @test infer_shape(affine, x, W, b) == (4, 20)
end