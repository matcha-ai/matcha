# Passes

> `engine::Pass` - alias for `std::function<void(Lambda&)>`

A function that passes through a `Lambda` and modifies it or otherwise
accesses it is called a `Pass`. Their implementation is often 
non-trivial, especially for optimization purposes. Passes 
are usually called by Transforms such as with JIT to simplify lambdas. 
However, e.g. the Matcha backpropagation system, which itself may be
formally called a pass (as it accepts a lambda and extends it by
gradient flow) calls some passes too for the backpropagation to be easier
and more modular. Passing through a valid lambda should leave it valid.


- `engine::debug(const Lambda& lambda) -> void` - prints the lambda and debugging info
  for the lambda, performs some health checks (e.g. topological order, flow integrity)
- `engine::init(Lambda& lambda) -> void` - initializes all operations in the lambda
- `engine::deadCodeElimination(Lambda& lambda) -> void` - prunes tensors and operations
  that no output or side effect depends on
- `engine::inlineExpansion(Lambda& lambda) -> void` - finds Lambdas nested inside the lambda,
  clones them, and recursively flattens them
- `engine::copyPropagation(Lambda& lambda) -> void` - removes _unnecessary_ identity
  operations (some identity operations are needed for side-effects)
- `engine::constantPropagation(Lambda& lambda) -> void` - runs all operations that do not
  depend on a non-constant value and prunes them
