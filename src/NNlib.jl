module NNlib

# Include our Activation functions
include("activations.jl")

# Get normalization like batchnorm
include("normalization.jl")

# Conv2D and friends
include("convolutional.jl")


end # module
