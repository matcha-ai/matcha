# Lambda
> `class engine::Lambda final`

Lambda is a sequence of `engine::Op` operations stored for lazy execution.
It is a fundamental class within the Matcha engine, since it mediates
many core functionalities of matcha, from 
[automatic differentiation](tensor/autograd) to [JIT](tensor/jit).

A lambda has its inputs and outputs, similarly to a single `engine::Op`.
Apart from inputs, lambda also has its constants, which are tensors needed
to produce the lambda's outputs that don't depend on any other operation
and are guaranteed not to change. Similar to them are side-inputs,
which don't depend on any operation within the lambda, but can be
modified externally.

Given all inputs, side-inputs, and constants, running a lambda
produces its outputs and side effects. For running lambdas, use
[Executors](engine/lambda/executors). However, it is common to
modify the lambda in some way (e.g. simplify it) before running it.
A higher-order function operating on lambdas is called a `Pass`. 
Matcha provides several pass functions for functionally simplifying 
lambdas or adding functionalities to them. 
[Read more](engine/lambda/passes).

## Public members

- `std::vector<Op*> ops` - stored operations (in topological order)
- `std::vector<Tensor*> tensors` - stored tensors
- `std::vector<Tensor*> inputs` - lambda inputs
- `std::vector<Tensor*> outputs` - lambda outputs
- `std::map<Tensor*, const tensor*> side_inputs` - side inputs

## Creating a lambda

For most purposes, refer to [tracing](engine/lambda/tracing). To create
a lambda manually, populate its public members, strictly following these rules:

- `ops` must be topologically sorted (upstream operations first, downstream operations last)
- `tensors` are given one extra `req()` for their presence within the lambda
- `ops` and `tensors` elements must be unique (i.e. they don't repeat)
- `ops` and `tensors` elements must not be present in another lambda, including clones
- all `inputs` and `outputs` are included in `tensors` (`outputs` must not be unique themselves, however)
- source `const tensor` pointers in `side_inputs` are expected to be valid
  for the entire lambda lifetime

To help check lambda validity, use the `engine::check` [pass](engine/lambda/passes.md).
