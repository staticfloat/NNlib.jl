@testset "linear_map" begin
    x = randn(10)
    W = randn(6, 10)
    @test linear_map(x, W) == W * x
    @test infer_shape(linear_map, x, W) == (6,)
end

@testset "linear_map backprop" begin
    # Starting from a random W, learn a "ground truth" W that is just eye(4)
    W = randn(4, 4)
    W_gt = eye(4)

    # Helper function to calculate the error of a given input
    function error(x)
        y = linear_map(x, W)
        y_gt = linear_map(x, W_gt)
        return y - y_gt
    end

    # Helper function to take a single learning step
    function learning_step(x, learning_rate=0.1)
        Δ = learning_rate .* error(x)
        dx, dW = ∇linear_map(Δ, x, W)
        W .-= dW
    end

    # Calculate the initial error for an input
    x = randn(4)
    e1 = error(x)

    # Take a single step forward
    learning_step(x)

    # Ensure that we actually just improved our error:
    @test norm(error(x)) < norm(e1)

    # Run many more iterations, and ensure we are within a reasonable bound:
    for idx in 1:100
        learning_step(randn(4))
    end
    @test norm(error(x)) < norm(e1)
    @test norm(W .- W_gt) < 1e-2
end

@testset "affine" begin
    x = randn(10)
    W = randn(6, 10)
    b = randn(1)
    @test affine(x, W, b) == W * x .+ b
    @test infer_shape(affine, x, W, b) == (6,)
end

@testset "affine backprop" begin
    # Starting from a random W, learn a "ground truth" W that is just eye(4)
    W = randn(4, 4)
    W_gt = eye(4)
    b = randn(4)
    b_gt = ones(4)

    # Helper function to calculate the error of a given input
    function error(x)
        y = affine(x, W, b)
        y_gt = affine(x, W_gt, b_gt)
        return y - y_gt
    end

    # Helper function to take a single learning step
    function learning_step(x, learning_rate=0.1)
        Δ = learning_rate .* error(x)
        dx, dW, db = ∇affine(Δ, x, W, b)
        W .-= dW
        b .-= db
    end

    # Calculate the initial error for an input
    x = randn(4)
    e1 = error(x)

    # Take a single step forward
    learning_step(x)

    # Ensure that we actually just improved our error:
    @test norm(error(x)) < norm(e1)

    # Run many more iterations, and ensure we are within a reasonable bound:
    for idx in 1:100
        learning_step(randn(4))
    end
    @test norm(error(x)) < norm(e1)
    @test norm(W .- W_gt) < 1e-2
    @test norm(b .- b_gt) < 1e-2
end