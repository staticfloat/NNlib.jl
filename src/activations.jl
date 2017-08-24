# Various activation functions
@multiexport sigmoid, σ, relu, elu, selu, softmax

# Sigmoid function
function σ!(out, xs)
    l = one(eltype(xs))
    out .= l ./ (l .+ exp.(-xs))
end
@outplace σ(xs)
function ∇σ!(out, Δ, xs)
    s = σ(xs)
    l = one(eltype(xs))
    out[1] .= Δ .* s .* (l - s)
    return out
end
@outplace ∇σ(Δ, xs)
# Also provide `sigmoid(x)` if the user doesn't like unicode as much
sigmoid!(out, xs) = σ!(out, xs)
@outplace sigmoid(xs)
∇sigmoid!(out, Δ, xs) = ∇σ!(out, Δ, xs)
@outplace ∇sigmoid(Δ, xs)


# REctified Linear Unit
function relu!(out, xs)
    out .= xs .* (xs .> zero(eltype(xs)))
end
@outplace relu(xs)
function ∇relu!(out, Δ, xs)
    out[1] .= Δ .* (xs .> zero(eltype(xs)))
    return out
end
@outplace ∇relu(Δ, xs)


# Exponential Linear Unit
function elu!(out, xs, α=one(eltype(xs)))
    select = xs .> 0
    l = one(eltype(xs))
    out .= xs .* select + α .* (exp.(xs) - l) .* .~select
end
@outplace elu(xs, α=one(eltype(xs)))
function ∇elu!(out, Δ, xs, α=one(eltype(xs)))
    select = xs .> 0
    l = one(eltype(xs))
    out[1] .= Δ .* l .* select + α .* (exp.(xs) - l) .* .~select
    return out
end
@outplace ∇elu(Δ, xs, α=one(eltype(xs)))


# Scaled Exponential Linear Unit (default α and λ to the values from the paper)
function selu!(out, xs, α=eltype(xs)(1.67326), λ=eltype(xs)(1.0507))
    select = xs .> 0
    l = one(eltype(xs))
    out .= λ .* (xs .* select + α .* (exp.(xs) - l) .* .~select)
end
@outplace selu(xs, α=eltype(xs)(1.67326), λ=eltype(xs)(1.0507))
function ∇selu!(out, Δ, xs, α=eltype(xs)(1.67326), λ=eltype(xs)(1.0507))
    select = xs .> 0
    l = one(eltype(xs))
    out[1] .= Δ .* λ .* (l .* select + α .* (exp.(xs) - l) .* .~select)
    return out
end
@outplace ∇selu(Δ, xs, α=eltype(xs)(1.67326), λ=eltype(xs)(1.0507))


# Softmax activation, summing across first axis if given Matrix
function softmax!(out, xs)
    out .= exp.(xs) ./ sum(exp, xs, 1)
end
@outplace softmax(xs)
function ∇softmax!(out, Δ, xs)
    s = sum(exp, xs, 1)
    out[1] .= exp.(xs)./s.*(Δ .- sum(Δ .* exp.(xs), 1)./s)
    return out
end
@outplace ∇softmax(Δ, xs)