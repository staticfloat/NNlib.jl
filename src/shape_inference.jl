## Shape inference; welcome to the most exciting part of this package.
export infer_shape

"""
infer_shape(func::DataType, args...)

Given an NNlib function, if called with `args...` return the shape of the
output.  This must be overridden for all functions that mutate the shape of
their inputs. If a function does not change shape (i.e. most activation
functions) the default `infer_shape()` implementation will suffice, which is
to simply return the shape of the primary input to the function (by convention,
this will be one of the first three arguments, depending on whether the
function is inplace or a gradient calculation function)

Example of overriding default shape inference for the `affine()` function:

    function infer_shape(::typeof(affine), xs, W, b)
        return (size(W, 1), size(xs, 2))
    end

Note that applying `infer_shape()` to an inplace function such as `affine!()`
does not typically make sense, as one of the arguments to `affine!()` is the
output array itself; to determine the size of the output array you must use,
apply `infer_shape()` to the outplace function, i.e. `affine()`.
"""
function infer_shape(func::Function, args...)
    # We first infer (har har) whether this function is an inplace function
    # by looking for an `!` at the end of its name.  If it is, we need to skip
    # the `out` parameter in front.  Furthermore, if it's a `∇` function, we
    # must skip the `Δ` argument as well.
    
    arg_idx = 1
    fname = String(Base.function_name(func))
    if fname[end] == '!'
        # Skip `out`
        arg_idx += 1
    end
    if fname[1] == '∇'
        # Skip `Δ`
        arg_idx += 1
    end

    # Assert that we're not about to push past the end of `args`
    assert_fail_msg = "Shape inference error!  Too few arguments to $(fname)!"
    @assert arg_idx <= length(args) assert_fail_msg

    # Return the shape of the primary input (usually called `xs`)
    return size(args[arg_idx])
end