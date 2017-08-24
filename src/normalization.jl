# Normalization functions like batchnorm
@multiexport batchnorm

# Batch normalization
function batchnorm!(out, xs, μ, σ, β, γ)
    out .= γ .* (xs .- μ) ./ sqrt(σ .+ eps(eltype(σ))) .+ β
end
@outplace batchnorm(xs, μ, σ, β, γ)

function ∇batchnorm!(out, Δ, xs, μ, σ, β, γ)
    s = (σ + eps(eltype(σ)))^(-1/2)
    e = Δ .* γ
    xd = xs .- μ

    # dx
    out[1] .= Δ .* (γ .* s)

    # dμ
    out[2] .= -sum_to_rank(e .* (one(eltype(s)) / s), ndim(μ))
    
    # dσ
    out[3] .= sum_to_rank(e .* xd .* -0.5 .* s^3, ndim(σ))
    
    # dβ
    out[4] .= sum_to_rank(Δ, ndim(β))
    
    # dγ
    out[5] .= sum_to_rank(Δ .* (xd .* s), ndim(γ))

    return out
end
@outplace ∇batchnorm(Δ, xs, μ, σ, β, γ)