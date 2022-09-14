# Passes

> `using engine::Pass = std::function<void(Lambda&)>`

A higher-order function that passes through a `Lambda` and modifies it 
or otherwise accesses it is called a `Pass`. Their implementation is often 
non-trivial, especially for optimization purposes. Passes 
are usually invoked by Transforms such as [JIT](tensor/jit).
The Matcha automatic differentiation system, which may be
formally called a pass too (as it accepts a lambda and extends it by
gradient flow) also calls some passes to make the task simpler
and more modular. 

!> Unless said explicitly otherwise, 
   passes assume the given lambda to be valid, strictly according to
   [these](engine/lambda/README#creating-a-lambda) specifications.
   Passing through a valid lambda should leave it valid.


- [`engine::check(const Lambda& lambda) -> void`](engine/lambda/passes/check) -
  runs some lambda validity checks:
- [`engine::constantPropagation(Lambda& lambda) -> void`](engine/lambda/passes/constant-propagation) - 
  runs deterministic non-side-effect operations depending on constant 
  tensors only and prunes them
- [`engine::copyPropagation(Lambda& lambda) -> void`](engine/lambda/passes/copy-propagation) - 
  contracts identity operations
- [`engine::deadCodeElimination(Lambda& lambda) -> void`](engine/lambda/passes/dead-code-elimination) - 
  prunes tensors and operations that no output or side effect depends on
- [`engine::debug(const Lambda& lambda) -> void` - ](engine/lambda/passes/debug)
  prints the lambda and related debugging info, runs `engine::check` and
  warns if non-zero value is returned
- [`engine::init(Lambda& lambda) -> void`](engine/lambda/passes/init) - 
  initializes operations in the lambda
- [`engine::inlineExpansion(Lambda& lambda) -> void`](engine/lambda/passes/inline-expansion) - 
  finds nested lambdas and recursively inlines them
- [`engine::matmulFusion(Lambda& lambda) -> void`](engine/lambda/passes/matmul-fusion) - fuses matrix 
   multiplications with adjacent transpose operations into a single operation

