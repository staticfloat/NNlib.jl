@testset "batchnorm" begin
    x = randn(10000)
    μ = mean(x)
    σ = var(x)
    β = 1.0
    γ = 2.0
    y = batchnorm(x, μ, σ, β, γ)

    # Ensure that this carefully-crafted batchnorm example correctly sets the
    # mean and variance we would expect from someone of his upbringing.
    @test abs(mean(y[:]) - 1.0) < 1e-4
    @test abs(var(y[:]) - 4.0) < 1e-4

    # Ensure that we can operate with very different sizes and broadcasting
    # "just works":
    x = randn(100, 20)
    μ = 0.0
    σ = abs.(randn(1, 20))
    β = 2.0*ones(1, 1)
    γ = randn(100, 1)
    y = batchnorm(x, μ, σ, β, γ)

    # Test the broadcasting is workign properly
    @test size(y) == (100, 20)
end