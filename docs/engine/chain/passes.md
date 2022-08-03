# Passes

> `engine::Pass` - alias for `std::function<void(Chain&)>`

A function that passes through a `Chain` and modifies it or otherwise
accesses it is called a `Pass`. Their implementation is often 
non-trivial, especially for optimization purposes. Passes 
are usually called by Decorators such as with JIT to simplify chains. 
However, e.g. the Matcha backpropagation system, which itself may be
formally called a pass (as it accepts a chain and extends it by
gradient flow) calls some passes too for the backpropagation to be easier
and more modular. Passing through valid chain should leave it valid.


- `engine::debug(Chain& chain) -> void` - prints the chain and debugging info
  for the chain, performs some health checks (e.g. topological order, flow integrity)
- `engine::reduceToEffects(Chain& chain) -> void` - prunes tensors and operations
  that no output or side effect depends on
- `engine::contractIdentities(Chain& chain) -> void` - removes _unnecessary_ identity
  operations (some identity operations are needed for side-effects)
- `engine::flatten(Chain& chain) -> void` - finds Chains nested inside the chain,
  clones them, and recursively flattens them
- `engine::foldConstants(Chain& chain) -> void` - runs all operations that do not
  depend on a non-constant value and prunes them
- `engine::initialize(Chain& chain) -> void` - initializes all operations in the chain
