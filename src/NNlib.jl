module NNlib

# First, our utility functions
include("util.jl")

# Include the various types of functions/layers we're interested in
include("activations.jl")
include("linear.jl")
include("normalization.jl")
include("convolutional.jl")
end # module
