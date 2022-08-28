# Softmax
> `nn::softmax(const tensor&) -> tensor` \
> `struct nn::Softmax`

Applies the [softmax](tensor/operations/miscellaneous#softmax)
activation function to the input batch, along the last dimension:

$ softmax(x) = \frac{e^{\odot \hat{x}}}{\sum_{i \in \hat{x}} e^i } $

?> Being an activation function, `softmax` is supported as a flag
   for available abstract layers, like the [`nn::Fc`](nn/layers/fc).
