@testset "batchnorm" begin
    x = randn(10000)
    μ = mean(x)
    σ = var(x)
    β = 1.0
    γ = 2.0
    y = batchnorm(x, μ, σ, β, γ)

    # Ensure that this carefully-crafted batchnorm example correctly sets the
    # mean and variance we would expect from someone of his upbringing.
    @test abs(mean(y) - 1.0) < 1e-7
    @test abs(var(y) - 4.0) < 1e-7
end