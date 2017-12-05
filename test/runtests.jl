using NNlib
using Base.Test

# Seed random numbers so as to not get random failures
srand(1337)

#include("util.jl")
#include("shape_inference.jl")
#include("activations.jl")
#include("linear.jl")
#include("normalization.jl")
include("convolutional.jl")