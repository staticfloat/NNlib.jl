# Linear layers such as the affine layer
@multiexport affine

function affine!(out, xs, W, b)
    out .= W * xs .+ b
end
@outplace affine(xs, W, b)
function ∇affine!(out, Δ, xs, W, b)
    # TODO: How do we backprop W and b?
    out .= Δ * W'
end
@outplace ∇affine(Δ, xs, W, b)