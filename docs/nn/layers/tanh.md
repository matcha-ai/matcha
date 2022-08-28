# Sigmoid
> `nn::tanh(const tensor&) -> tensor` \
> `struct nn::Tanh`

Applies the [hyperbolic tangent](tensor/operations/elementwise-unary#tanh)
activation function to the input batch:

$ tanh(x) = 2 \sigma(2 x) - 1$

where

$ \sigma(x) = \frac{1}{1 + e^{-x}} $


?> Being an activation function, `tanh` is supported as a flag
   for available abstract layers, like the [`nn::Fc`](nn/layers/fc).
