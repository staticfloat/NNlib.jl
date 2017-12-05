# Linear functions such as the linear map, or an affine transformation
@multiexport affine, linear_map

"""
A linear map expresses each output as a linear combination of all inputs;
computationally, it is commonly represented as a matrix multiply.
"""
function linear_map!(out, xs, W)
    out .= W * xs
end
@outplace linear_map(xs, W)
function ∇linear_map!(out, Δ, xs, W)
    # dx
    out[1] .= W' * Δ
    
    # dW
    out[2] .= Δ * xs'

    return out
end
@outplace ∇linear_map(Δ, xs, W)

# We need to explicitly define shape inference for linear_map:
function infer_shape(::typeof(linear_map), xs, W)
    # Ensure that the young hooligans do not try to shove huge tensors in
    assert_msg = "Cannot apply $(ndims(xs))-rank tensor to linear_map()!"
    @assert ndims(xs) in [1, 2] assert_msg

    if ndims(xs) == 1
        return (size(W, 1),)
    else #ndims(xs) == 2
        return (size(W, 1), size(xs, 2))
    end
end


"""
An affine transformation expresses each output as a linear combination of all
inputs with a separate bias term added to each output.
"""
function affine!(out, xs, W, b)
    out .= W * xs .+ b
end
@outplace affine(xs, W, b)
function ∇affine!(out, Δ, xs, W, b)
    # dx
    out[1] .= W' * Δ
    
    # dW
    out[2] .= Δ * xs'

    # db
    out[3] .= Δ

    return out
end
@outplace ∇affine(Δ, xs, W, b)

# We need to explicitly define shape inference for affine:
function infer_shape(::typeof(affine), xs, W, b)
    # Ensure that the young hooligans do not try to shove huge tensors in
    assert_msg = "Cannot apply $(ndims(xs))-rank tensor to affine function!"
    @assert ndims(xs) in [1, 2] assert_msg

    if ndims(xs) == 1
        return (size(W, 1),)
    else #ndims(xs) == 2
        return (size(W, 1), size(xs, 2))
    end
end