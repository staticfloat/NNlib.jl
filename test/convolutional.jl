@testset "conv2d" begin
    # Input is random noise representing a "32x32 RGB image"
    x = randn(32, 32, 3)

    # Our first kernel will be a 5x5 kernel that outputs 1 channel, but really
    # it's just a passthrough of the 2nd channel of the input:
    W = zeros(1, 5, 5, size(x, 3))
    W[1, 3, 3, 2] = 1.0

    # Test to show that this output is equivalent to the input's 2nd channel
    y = conv2d(x, W, :SAME)
    @test size(y) == size(x[:,:,2:2])
    @test y ≈ x[:,:,2]

    # Next, do this with :VALID convolution size:
    hks_x = div(size(W,3),2)
    hks_y = div(size(W,2),2)
    y = conv2d(x, W, :VALID)
    @test size(y) == size(x[1+hks_y:end-hks_y,1+hks_x:end-hks_x,2:2])
    @test y ≈ x[1+hks_y:end-hks_y,1+hks_x:end-hks_x,2]

    # Next, bump our kernel up to do a little 3x3 summation:
    W[1, 2:4, 2:4, 2] = 1.0
    y = conv2d(x, W, :SAME)

    # Now check the corners to ensure that output is what we'd expect
    @test y[1, 1, 1] ≈ sum(x[1:2, 1:2, 2])
    @test y[1, 2, 1] ≈ sum(x[1:2, 1:3, 2])
    @test y[2, 2, 1] ≈ sum(x[1:3, 1:3, 2])
    @test y[3, 2, 1] ≈ sum(x[2:4, 1:3, 2])
    @test y[3, 3, 1] ≈ sum(x[2:4, 2:4, 2])
    @test y[end, 1, 1] ≈ sum(x[end-1:end, 1:2, 2])
    @test y[1, end, 1] ≈ sum(x[1:2, end-1:end, 2])
    @test y[end, end, 1] ≈ sum(x[end-1:end, end-1:end, 2])
end