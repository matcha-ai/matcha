# Operations

Tensor operations are a pivotal component of Matcha. Matcha implements
most common functions in a way that is convenient to use. To make Matcha
easy to use within the already present ecosystem of libraries, the behavior
of Matcha operations in most cases copies
[Numpy](https://numpy.org/doc/1.23/) or 
[TensorFlow](https://www.tensorflow.org/api_docs/python/tf). You can refer
to their appropriate docs if not covered enough here yet.

## Hierarchy

For easy understanding and implementation, Matcha operations build 
on a few abstract base operations, whenever possible:

- [Elementwise unary](tensor/operations/elementwise-unary) operations
- [Elementwise binary](tensor/operations/elementwise-binary) operations
- [Reduction](tensor/operations/reduction) operations

... and [miscellaneous](tensor/operations/miscellaneous) operations


## Custom operations

To create custom operations, refer to the backend engine
[Op](engine/op/), [Tensor](engine/tensor/) and [kernels](engine/kernels)
([custom example](engine/op/example)).
