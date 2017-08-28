@testset "zeropad" begin
    x = randn(2, 2, 2)
    xz = zeropad(x, [(0, 0), (1, 1), (0, 2)])

    @test size(xz) == (2, 4, 4)
    @test all(xz[:,1,:] .== 0)
    @test all(xz[:,end,:] .== 0)
    @test all(xz[:,:,end-1] .== 0)
    @test all(xz[:,:,end] .== 0)
end