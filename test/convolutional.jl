@testset "conv2d" begin
    # Input is random noise representing a "32x32 RGB image"
    x = randn(1, 32, 32, 3)

    # Our first kernel will be a 5x5 kernel that outputs 1 channel, but really
    # it's just a passthrough of the 2nd channel of the input:
    W = zeros(1, 5, 5, size(x, 4))
    W[1, 3, 3, 2] = 1.0

    # Test to show that this output is equivalent to the input's 2nd channel
    y = conv2d(x, W, :SAME)
    @test size(y) == size(x[:,:,:,2:2])
    @test y ≈ x[:,:,:,2]

    # Next, do this with :VALID convolution size:
    hks_x = div(size(W,3),2)
    hks_y = div(size(W,2),2)
    y = conv2d(x, W, :VALID)
    @test size(y) == size(x[:,1+hks_y:end-hks_y,1+hks_x:end-hks_x,2:2])
    @test y ≈ x[:,1+hks_y:end-hks_y,1+hks_x:end-hks_x,2]

    # Next, bump our kernel up to do a little 3x3 summation:
    W[1, 2:4, 2:4, 2] = 1.0
    y = conv2d(x, W, :SAME)

    # Now check the corners to ensure that output is what we'd expect
    @test y[1, 1, 1, 1] ≈ sum(x[1, 1:2, 1:2, 2])
    @test y[1, 1, 2, 1] ≈ sum(x[1, 1:2, 1:3, 2])
    @test y[1, 2, 2, 1] ≈ sum(x[1, 1:3, 1:3, 2])
    @test y[1, 3, 2, 1] ≈ sum(x[1, 2:4, 1:3, 2])
    @test y[1, 3, 3, 1] ≈ sum(x[1, 2:4, 2:4, 2])
    @test y[1, end, 1, 1] ≈ sum(x[1, end-1:end, 1:2, 2])
    @test y[1, 1, end, 1] ≈ sum(x[1, 1:2, end-1:end, 2])
    @test y[1, end, end, 1] ≈ sum(x[1, end-1:end, end-1:end, 2])
end


@testset "conv2d backprop" begin
    # Starting from a random W, learn a "ground truth" W
    W = randn(1, 3, 3, 3)
    W_gt = randn(1, 3, 3, 3)

    # Helper norm() function
    function norm(x::Array{T,4}) where T
        return norm(flatten(x))
    end

    # Helper function to calculate the error of a given input
    function error(x)
        y = conv2d(x, W, :SAME)
        y_gt = conv2d(x, W, :SAME)
        return y - y_gt
    end

    # Helper function to take a single learning step
    function learning_step(x, learning_rate=0.1)
        Δ = learning_rate .* error(x)
        dx, dW = ∇conv2d(Δ, x, W, :SAME)
        W .-= dW
    end

    # Calculate the initial error for an input
    x = randn(1, 10, 10, 3)
    e1 = error(x)

    # Take a single step forward
    learning_step(x)

    # Ensure that we actually just improved our error:
    @test norm(error(x)) < norm(e1)

    # Run many more iterations, and ensure we are within a reasonable bound:
    #for idx in 1:100
    #    learning_step(randn(1, 10, 10, 3))
    #end
    #@test norm(error(x)) < norm(e1)
    #@test norm(W .- W_gt) < 1e-2
end