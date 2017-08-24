@multiexport conv2d

"""
Naive conv2d() implementation; just loop over kernels in W, directly performing
dot product.  Do not expect this to be a competitive formulation.  This is
included for testing purposes only.
"""
function conv2d!(out, xs, W, b, padding_mode=:SAME)
    # I'm too tired tonight, let's try this tomorrow.
end
@outplace conv2d(xs, W, b, padding_mode=:SAME)

function infer_shape(::typeof(conv2d), xs, W, b, padding_mode=:SAME)
end