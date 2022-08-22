# Lambda

> `engine::Lambda`

Lambda is a sequence of `engine::Op` operations stored for lazy execution.
It is a fundamental class within the Matcha engine, since it mediates
many core functionalities of matcha, from 
[automatic differentiation](tensor/autograd) to [JIT](tensor/jit).

A lambda has its inputs and outputs, similarly to a single `engine::Op`.
Apart from inputs, lambda also has its constants, which are tensors needed
to produce the lambda's outputs that don't depend on any other operation
and are guaranteed not to change. Similar to them are side-variables,
which don't depend on any operation within the lambda, but can mutate.

Given all inputs, constants, and side-variables, running a lambda
produces its outputs and side effects. For running lambdas, use
[Executors](engine/lambda/executors). However, it is common to
modify the lambda in some way (e.g. simplify it) before running it.
A function that e.g. modifies a lambda is called a `Pass`. Matcha provides
several pass functions for functionally simplifying lambdas or adding functionalities
to them. [Read more](engine/lambda/passes).

## Public members

- `std::vector<Op*> ops` - stored operations (in topological order)
- `std::vector<Tensor*> tensors` - stored tensors
- `std::vector<Tensor*> inputs` - lambda inputs
- `std::vector<Tensor*> outputs` - lambda outputs

## Creating a lambda

For most purposes, refer to [tracing](engine/lambda/tracing). To create
a lambda manually, populate its public members, strictly following these rules:

- `ops` must be topologically sorted
- `ops` must not share any operation with another lambda
- `tensors` contents don't repeat (i.e. one tensor is not mentioned twice)
- `tensors` are given one extra `req()` for their presence within the lambda
- all `inputs` and `outputs` are included in `tensors`


## Basic tools

- `engine::clone(const Lambda& lambda) -> Lambda` - creates a clone of the original lambda,
  the clone has the same effects and supported side-effects as the original
- `operator<<(std::ostream& os, const Lambda& lambda) -> std::ostream&` - prints lambda representation into the stream
