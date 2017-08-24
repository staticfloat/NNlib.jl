# NNlib

## Design decisions

We implement a primarily Functional API; no state carrying objects to hold weights, etc...  Pass in all necessary arguments directly to all functions.  Ordering follows the convention of `output`, `input`, `parameters`, as exemplified by the `affine!()` function definition:

```
function affine!(out, xs, W, b)
    out .= W * xs .+ b
end
```

Note that non-inplace (often referred to in the code as `outplace`) methods are available without the leading `!`, e.g. `affine(xs, W, b)`.  These functions use shape inference (see `infer_shape()` for details on how that works) to build an array to hold the result of the computation, then call the inplace method on that new array.

Gradient computation functions are available by prepending `∇` to the function name, e.g. `∇affine!(out, Δ, xs, W, b)` will calculate the gradient for the above-defined `affine!()` method.  Note that all gradient computations return one gradient per input parameter, e.g. `∇affine!` will expect `out` to be a three-tuple consisting of `(Δxs, ΔW, Δb)`, and these will be assigned to inplace.  If a parameter is optional (this denotes some kind of parameter that should not be considered for gradient calculation) it is not included within this list.

## Things I am purposefully not thinking about

* I'm not thinking about how to do shape inference for things like passing in a bias with a shape of `(10, 1, 1, 1)` and having to worry about broadcasting automagically jumping a matrix up to a 4-d tensor.