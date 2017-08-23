# Normalization functions like batchnorm
@multiexport batchnorm

# Batch normalization
function batchnorm!(out, x, μ, σ, β, γ)
    out .= γ.*(x .- μ)./sqrt(σ .+ eps(eltype(σ))) .+ β
end
@outplace batchnorm(x, μ, σ, β, γ)

function ∇batchnorm!(out, Δ, x, μ, σ, β, γ)
    # TODO: How do we backprop out stuff for μ, σ, β, γ?
    out .= Δ .* (γ .* sqrt(σ + eps(eltype(σ))))
end
@outplace ∇batchnorm(Δ, x, μ, σ, β, γ)