# Chain

> `engine::Chain`

Chain is a sequence of `engine::Op` operations stored for lazy execution.
It is a fundamental class within the Matcha engine, since it mediates
many core functionalities of matcha, from 
[automatic differentiation](tensor/autograd) to [JIT](tensor/jit).

A chain has its inputs and outputs, similarly to a single `engine::Op`.
Apart from inputs, chain also has its constants, which are tensors needed
to produce the chain's outputs that don't depend on any other operation
and are guaranteed not to change. Similar to them are side-variables,
which don't depend on any operation within the chain, but can mutate.

Given all inputs, constants, and side-variables, running a chain
produces its outputs and side effects. For running chains, use
[Executors](engine/chain/executors). However, it is common to
modify the chain in some way (e.g. simplify it) before running it.
A function that modifies a chain is called a `Pass`. Matcha provides
several pass functions for functionally simplifying chains or adding functionalities
to them. [Read more](engine/chain/passes).

## Public members

- `std::vector<Op*> ops` - stored operations (in topological order)
- `std::vector<Tensor*> tensors` - stored tensors
- `std::vector<Tensor*> inputs` - chain inputs
- `std::vector<Tensor*> outputs` - chain outputs

## Creating a chain

For most purposes, refer to [tracing](engine/chain/tracing). To create
a chain manually, populate its public members, strictly following these rules:

- `ops` must be topologically sorted
- `ops` must not share any operation with another chain
- `tensors` contents don't repeat (i.e. one tensor is not mentioned twice)
- `tensors` are given one extra `req()` for their presence within the chain
- all `inputs` and `outputs` are included in `tensors`


## Basic tools

- `engine::clone(const Chain& chain) -> Chain` - creates a clone of the original chain,
  the clone has the same effects and supported side-effects as the original
- `operator<<(std::ostream& os, const Chain& chain) -> std::ostream&` - prints chain representation into the stream
