# ReLU
> `nn::relu(const tensor&) -> tensor` \
> `struct nn::Relu`

Applies the Rectified Linear Unit (ReLU) activation function
to the input batch. Equivalent to calling:

```cpp
tensor output = maximum(input, 0);
```

?> Being an activation function, `relu` is supported as a flag
   for available abstract layers, like the [`nn::Fc`](nn/layers/fc).
