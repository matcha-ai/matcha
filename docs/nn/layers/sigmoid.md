# Sigmoid
> `nn::sigmoid(const tensor&) -> tensor` \
> `struct nn::Sigmoid`

Applies the [sigmoid](tensor/operations/elementwise-unary#sigmoid)
activation function to the input batch:

$ \sigma(x) = \frac{1}{1 + e^{-x}} $

?> Being an activation function, `sigmoid` is supported as a flag
   for available abstract layers, like the [`nn::Fc`](nn/layers/fc).
