# Normalization functions like batchnorm
export batchnorm, ∇batchnorm

# Batch normalization
batchnorm(x, μ, σ, β, γ) = γ.*(x .- μ)./sqrt(σ .+ eps(eltype(σ)) .+ β

function ∇batchnorm(Δ, x, μ, σ, β, γ)
    # bleep bloop
end
