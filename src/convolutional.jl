@multiexport conv2d

"""
Naive 2d convolution implementation; just loop over kernels in W, directly
performing dot product.  Do not expect this to be a competitive formulation.
This is included for testing and completeness purposes.

Dimensional ordering of input is  [y, x, in_chan]
Dimensional ordering of kernel is [out_chan, y, x, in_chan]
"""
function conv2d!(out, xs, W, padding_mode=:SAME)
    # The "half-kernel-shape" is an important dimension to know
    hks_x = div(size(W,3),2)
    hks_y = div(size(W,2),2)

    # Calculate the domain over which we will calculate `out`
    start_x = 1
    start_y = 1
    stop_x = size(xs,2) - 2*hks_x
    stop_y = size(xs,1) - 2*hks_y

    if padding_mode == :SAME
        stop_x += 2*hks_x
        stop_y += 2*hks_y

        # Pad `xs` up by a half-kernel in every spatial direction
        size_pad = collect(size(xs))
        size_pad[1] += 2*hks_y
        size_pad[2] += 2*hks_x
        xs_pad = zeros(eltype(xs), size_pad...)
        xs_pad[1+hks_y:end-hks_y, 1+hks_x:end-hks_x,:] = xs
        xs = xs_pad
    end

    k_yl = size(W,3)
    k_xl = size(W,2)
    # For each kernel
    for kidx in 1:size(W,1)
        # Iterate over every output pixel
        for y in 1:(stop_y - start_y + 1)
            for x in 1:(stop_x - start_x + 1)
                # Grab the relevant slice of `xs`
                xs_idxs = (
                    # First, y indices, starting at `start_y`, and walking
                    # forward with `y`, of width `k_yl`
                    start_y + y - 1 : start_y + y + k_yl - 2,
                    # Same for x
                    start_x + x - 1: start_x + x + k_xl - 2,
                    # we always sum across all channels
                    :
                )

                # Perform dot product between that slice of `xs` and our
                # kernels, storing the result into `out`
                out[y, x, kidx] .= sum(W[kidx,:,:,:] .* xs[xs_idxs...])
            end
        end
    end

    return out
end
@outplace conv2d(xs, W, padding_mode=:SAME)

function infer_shape(::typeof(conv2d), xs, W, padding_mode=:SAME)
    shape = collect(size(xs))

    # New number of channels is the number of output channels in the kernel
    shape[3] = size(W, 1)

    # If we are doing "valid" padding, reduce the edges shapes
    if padding_mode == :VALID
        shape[1] -= (size(W,2) - 1)
        shape[2] -= (size(W,3) - 1)
    end

    return tuple(shape...)
end