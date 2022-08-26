# Neural network layers

Layers are a fundamental concept in artificial neural networks. Usually
it's inefficient in both semantic terms and performance-wise to talk
about individual neurons or scalar functions. Nobody wants
to enumerate millions of parameters, which modern neural networks
can easily have, by hand. Instead, a `Layer` represents
some family of such components and operations, which allows for
parallelization and greater representative capacity, and which can
be easily scaled and configured to our needs.


Layers usually behave as unary operators. They accept one `tensor` with
a batch of inputs, and return one `tensor`. This makes it simple for
you to create custom layers. However, Matcha implements commonly occuring
layers, see the following lists:


## Feed forward networks

- [`nn::flatten`, `class nn::Flatten`]() -
  flatten inputs to the shape `{batch_size, rest}`
- [`nn::relu`, `class nn::Relu`]() -
  the Rectangular Linear Unit (ReLU)
- [`nn::sigmoid`, `class nn::Sigmoid`]() -
  the [sigmoid](tensor/operations/elementwise-unary#sigmoid) function
- [`nn::tanh`, `class nn::Tanh`]() -
  the [tanh](tensor/operations/elementwise-unary#tanh) function
- [`nn::softmax`, `class nn::Softmax`]() -
  the [softmax](tensor/operations/miscellaneous#softmax) function
- [`class nn::Linear`]() - 
  perform stateful linear (affine) transformation to the inputs
- [`class nn::Fc`]() - 
  easily configurable wrapper for `nn::Linear` with various activation
  functions and normalizations

## Convolutional neural networks

Work in progress.

## Recurrent neural networks

Work in progress.

## Transformers

Work in progress.
