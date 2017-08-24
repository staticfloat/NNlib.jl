# Linear functions such as the affine, or linear map
@multiexport affine

function affine!(out, xs, W, b)
    out .= W * xs .+ b
end
@outplace affine(xs, W, b)
function ∇affine!(out, Δ, xs, W, b)
    # dx
    out[1] .= Δ * W'

    # dW
    out[2] .= xs' * Δ

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