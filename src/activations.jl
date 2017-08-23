# Various activation functions
@multiexport sigmoid, σ, relu, elu, softmax

# Sigmoid function
function σ!(out, xs)
    l = one(eltype(xs))
    out .= l ./ (l .+ exp.(-xs))
end
@outplace σ(xs)
function ∇σ!(out, Δ, xs)
    s = σ(xs)
    l = one(eltype(xs))
    out .= Δ .* s .* (l - s)
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
    out .= Δ .* (xs .> zero(eltype(xs)))
end
@outplace ∇relu(Δ, xs)



# Exponential Linear Unit
function elu!(out, xs, alpha)
    select = xs .> 0
    l = one(eltype(xs))
    out .= xs .* select + alpha .* (exp.(xs) - l) .* .~select
end
elu!(out, xs) = elu!(out, xs, one(eltype(xs)))
@outplace elu(xs, alpha)
@outplace elu(xs)
function ∇elu!(out, xs, alpha)
    select = xs .> 0
    l = one(eltype(xs))
    out .= l .* select + alpha .* (exp.(xs) - l) .* .~select
end
∇elu!(out, xs) = ∇elu!(out, xs, one(eltype(xs)))
@outplace ∇elu(out, xs, alpha)
@outplace ∇elu(out, xs)



# Softmax activation, summing across first axis if given Matrix
function softmax!(out, xs)
    out .= exp.(xs) ./ sum(exp, xs, 1)
end
@outplace softmax(xs)

function ∇softmax!(out, Δ, xs)
    s = sum(exp, xs, 1)
    out .= exp.(xs)./s.*(Δ .- sum(Δ .* exp.(xs), 1)./s)
end
@outplace ∇softmax(Δ, xs)