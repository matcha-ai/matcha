# Passes
> `"bits_of_matcha/engine/lambda/Pass.h"`\
> `using engine::Pass = std::function<void(Lambda&)>`

A higher-order function that passes through a [`Lambda`](engine/lambda/) and modifies it 
or otherwise accesses it is called a `Pass`. 

The implementation of passes is often 
non-trivial, especially for optimization purposes. Passes 
are usually invoked by Transforms such as the
[`JitTransform`](engine/transform/README#jittransform).
The Matcha automatic differentiation system, which may be
formally called a pass too (as it accepts a lambda and extends it by
gradient flow) also calls some passes, such as
[`engine::inlineExpansion`](engine/lambda/passes/inline-expansion),
to make the task simpler and more modular. 

!> Unless stated explicitly otherwise, 
   passes assume the given lambda to be valid, strictly according to
   [these](engine/lambda/README#creating-a-lambda) specifications.
   Passing through a valid lambda should leave it valid.
   To help check lambda validity, use the 
   [`engine::check`](engine/lambda/passes/check) pass. 
   For inspection, use [`engine::debug`](engine/lambda/passes/debug).


Details and examples are provided in separate articles:

- [`engine::check(const Lambda& lambda) -> void`](engine/lambda/passes/check) -
  runs automatic lambda validity checks
- [`engine::constantPropagation(Lambda& lambda) -> void`](engine/lambda/passes/constant-propagation) - 
  runs deterministic non-side-effect operations depending on constant 
  tensors only and prunes them
- [`engine::copyPropagation(Lambda& lambda) -> void`](engine/lambda/passes/copy-propagation) - 
  contracts identity operations
- [`engine::deadCodeElimination(Lambda& lambda) -> void`](engine/lambda/passes/dead-code-elimination) - 
  prunes tensors and operations that no output or side effect depends on
- [`engine::debug(const Lambda& lambda) -> void`](engine/lambda/passes/debug) -
  prints the lambda and related debugging info, runs 
  [`engine::check`](engine/lambda/passes/check) 
  and warns if non-zero value is returned
- [`engine::init(Lambda& lambda) -> void`](engine/lambda/passes/init) - 
  initializes operations in the lambda
- [`engine::inlineExpansion(Lambda& lambda) -> void`](engine/lambda/passes/inline-expansion) - 
  finds nested lambdas and recursively inlines them
- [`engine::matmulFusion(Lambda& lambda) -> void`](engine/lambda/passes/matmul-fusion) - fuses matrix 
   multiplications with adjacent transpose operations into a single operation

