# Various helpful macros and utilities
export zeropad

"""
Helper macro to generate non-inplace versions of functions.

As an example, given a function elu!(out, xs, α), call:

    @outplace elu(xs, α=1)

This will generate a function definition equivalent to:

    elu(xs, α=1) = elu!(similar(xs), xs, α)

Note that this macro actually uses shape inference to determine the correct
shape for the new `out` that is generated.  See `infer_shape()` for details on
shape inferrence and how that works/is defined.
"""
macro outplace(e)
    # Generate the inplace name which has the `!` at the end
    fname = esc(Symbol(e.args[1],:!))

    # Arguments can only be optional, we don't do kwargs.  So strip away the
    # right hand side of the default args when generating the actual call to
    # the inplace version of the function, otherwise Julia thinks we want to
    # set kwargs:
    function dekw(arg)
        if typeof(arg) <: Expr && arg.head == :kw
            return arg.args[1]
        end
        return arg
    end
    args = esc.(e.args[2:end])
    dekw_args = esc.(dekw.(e.args[2:end]))

    # Infer the shape and type for this invocation.  This is not knowabale in
    # general, and so we sub out to `infer_shape()`.  For forward pass, the
    # `infer_shape()` function defaults to an array of the same shape as the
    # first input argument, for backward pass, the default is a tuple of arrays
    # with each the same shape as its corresponding input.
    shape = :(infer_shape($(e.args[1]), $(dekw_args...)))
    t = :(eltype($(args[1])))
    new_out = :(similar($(args[1]), $(t), $(shape)))

    return quote
        $(esc(e.args[1]))($(args...)) = $(fname)($(new_out), $(dekw_args...))
    end
end

"""
Given function name `foo`, exports `foo`, `foo!`, `∇foo`` and `∇foo!`

Can also accept multiple function names, Example:
    
    @multiexport foo, bar

Will generate the code:

    export foo, foo!, ∇foo, ∇foo!, bar, bar!, ∇bar, ∇bar!
"""
macro multiexport(f)
    function gen_names(n)
        return [n, Symbol(n, :!), Symbol(:∇,n), Symbol(:∇,n,:!)]
    end

    # If we were given only a single thing to export, wrap it in a tuple
    if !(typeof(f) <: Expr)
        f = :(($(f),))
    end
        
    # If we've been given multiple things to export, do them all
    names = []
    for n in f.args
        push!(names, gen_names(n)...)
    end

    return Expr(:export, names...)
end


"""
Given an array `x` and a rank `r`, sums across dimensions in `x` until the
desired rank is achieved.  Higher dimensions are summed first, e.g. summation
proceeds from "left" to "right"; that is:

    sum_to_rank(randn(4, 3, 2, 1), 2)

Will result in a matrix of size `(2, 1)`.
"""
function sum_to_rank(x, r)
    @assert r >= 1 "Cannot sum down to a rank of $(r)!"
    while ndim(x) > r
        x = squeeze(sum(x, 1), 1)
    end
    return x
end


"""
Zeropad `x` with `sizes`, where `sizes` is a list of tuples denoting the number
of zeroes to insert before and after each axis.  Example:

    zeropad(randn(2,2,2), [(0, 2), (2, 1), (0, 0)])

will result in an output of shape `(4, 5, 2)`, with the last two slices of the
first dimension all being zero, and the first two and last one slices of the 
second dimension all being zero.
"""
function zeropad(x, sizes)
    # Calculate the total output size
    size_pad = collect(size(x))
    for idx in 1:length(sizes)
        size_pad[idx] += sizes[idx][1] + sizes[idx][2]
    end

    # Create zeroed-out placeholder
    x_pad = zeros(eltype(x), size_pad...)

    # Slap `x` into `x_pad`
    x_pad_idxs = (1+sizes[i][1] : sizes[i][1]+size(x,i) for i in 1:ndims(x))
    x_pad[x_pad_idxs...] = x

    # Return the padded result
    return x_pad
end