@multiexport conv2d

"""
Naive 2d convolution implementation; just loop over kernels in W, directly
performing dot product.  Do not expect this to be a competitive formulation.
This is included for testing and completeness purposes.

Dimensional ordering of input is  [batches, y, x, in_chan]
Dimensional ordering of kernel is [out_chan, y, x, in_chan]
"""
function conv2d!(out, xs, W, padding_mode=:SAME)
    # The "half-kernel-shape" is an important dimension to know
    hks_x = div(size(W,3),2)
    hks_y = div(size(W,2),2)

    # Calculate the domain over which we will calculate `out`
    start_x = 1
    start_y = 1
    stop_x = size(xs,3) - 2*hks_x
    stop_y = size(xs,2) - 2*hks_y

    if padding_mode == :SAME
        stop_x += 2*hks_x
        stop_y += 2*hks_y

        # Pad `xs` up by a half-kernel in every spatial direction
        size_pad = collect(size(xs))
        size_pad[2] += 2*hks_y
        size_pad[3] += 2*hks_x
        xs = zeropad(xs, [(0, 0), (hks_y, hks_y), (hks_x, hks_x), (0, 0)])
    end

    k_yl = size(W,3)
    k_xl = size(W,2)
    # For each batch
    for bidx in 1:size(xs, 1)
        # For each kernel
        for kidx in 1:size(W,1)
            # Iterate over every output pixel
            for y in 1:(stop_y - start_y + 1)
                for x in 1:(stop_x - start_x + 1)
                    # Grab the relevant slice of `xs`
                    x_idxs = (
                        # Grab just this one batch
                        bidx,
                        # Grab y indices starting at `start_y`, and walking
                        # forward with `y`, of width `k_yl`
                        start_y + y - 1 : start_y + y + k_yl - 2,
                        # Same for x
                        start_x + x - 1: start_x + x + k_xl - 2,
                        # we always sum across all channels
                        :
                    )

                    # Perform dot product between that slice of `xs` and our
                    # kernels, storing the result into `out`
                    out[bidx, y, x, kidx] = sum(W[kidx,:,:,:] .* xs[x_idxs...])
                end
            end
        end
    end

    return out
end
@outplace conv2d(xs, W, padding_mode=:SAME)

# For an explanation of this, check out TensorFlow's conv2d backprop comments:
# Search `https://github.com/tensorflow/tensorflow` for `conv_grad_ops.h`
function ∇conv2d!(out, Δ, xs, W, padding_mode=:SAME)
    # dx
    W_t = permutedims(W, [4, 2, 3, 1])[:,end:-1:1,end:-1:1,:]
    conv2d!(out[1], Δ, W_t, :SAME)

    # dW
    xs_t = permutedims(xs, [4, 2, 3, 1])[:,end:-1:1,end:-1:1,:]
    out_tmp = similar(out[2], eltype(out[2]), size(out[2])[[4, 2, 3, 1]])
    conv2d!(out_tmp, xs_t, Δ, :VALID)
    permutedims!(out[2], out_tmp, [4, 2, 3, 1])

    @show size(out[1])
    @show out[1]
    @show size(out[2])
    @show out[2]
    return out
end
@outplace ∇conv2d(Δ, xs, W, padding_mode=:SAME)

function infer_shape(::typeof(conv2d), xs, W, padding_mode=:SAME)
    shape = collect(size(xs))

    # New number of channels is the number of output channels in the kernel
    shape[4] = size(W, 1)

    # If we are doing "valid" padding, reduce the edges shapes
    if padding_mode == :VALID
        shape[2] -= (size(W,2) - 1)
        shape[3] -= (size(W,3) - 1)
    end

    return tuple(shape...)
end