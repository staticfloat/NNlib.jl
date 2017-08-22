# Various activation functions
export  sigmoid,  σ,  relu,  softmax,
       ∇sigmoid, ∇σ, ∇relu, ∇softmax

# Sigmoid function
σ(x) = 1 / (1 + exp(-x))
function ∇σ(Δ, x)
    s = σ(x)
    return Δ .* s .* (1 - s)
end

# Also provide `sigmoid(x)` if the user doesn't like unicode as much
sigmoid(x) = σ(x)
∇sigmoid(Δ, x) = ∇σ(Δ, x)

# REctified Linear Unit 
relu(x) = max(0, x)
∇relu(Δ, x) = Δ .* (x .> 0)

# Softmax activation, intelligently sum across first axis if given a matrix
softmax(xs::AbstractVector) = exp.(xs) ./ sum(exp.(xs))
softmax(xs::AbstractMatrix) = exp.(xs) ./ sum(exp.(xs), 1)
function ∇softmax(Δ, xs::AbstractVector)
    # Calculate Jacobian of softmax.  Refer to this SO answer for context:
    # https://stats.stackexchange.com/a/92309/23925
    ys = softmax(xs)
    J = [-ys[i]*ys[j] for i in 1:length(xs), j in 1:length(xs)]
    for i in 1:length(xs)
        J[i, i] += ys[i]
    end

    return J * Δ
end

function ∇softmax(Δ, xs::AbstractMatrix)
    for idx in 1:size(xs, 2)
        Δ[:, idx] = ∇softmax(Δ[:, idx], xs)
    end
    return Δ
end
