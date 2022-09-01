# JIT compilation
> `jit(const fn& function) -> fn`

JIT is an easy-to-use tool for transforming dynamic programs into
static ones. Roughly, this can be thought of as the difference
between writing a dynamic [PyTorch](https://pytorch.org/) code and 
building a static [TensorFlow](https://www.tensorflow.org/) graph.
Both have advantages and disadvantages. Dynamic code is easier to debug
and allows greater flexibility. Static graphs, on the other hand, are
completely language-agnostic and therefore very portable, and often allow
non-trivial optimizations for performance and memory. Matcha is designed
in a way that enables both frameworks and lets you decide which is better
for specific applications.


To JIT a function, simply call `jit`:

```cpp
tensor foo(tensor a) {
  // do lot of things to `a`
  return a;
}

int main() {
  auto fooButOptimized = jit(foo);

  tensor a = uniform(100, 100);
  tensor b = fooButOptimized(a);
  std::cout << b << std::endl;
}

```

## Limitations

As mentioned above, not every code can, or should, be JITed. Matcha makes
use of the [tracing](tensor/tracing) technique to inspect JITed functions.
This imposes several [limitations](tensor/tracing#limitations) on the JITed
code, such as:

- Inhibited IO side effects
- Ignoring flow control native to the programming language
  (like `if`, `while`, `switch`)
- Invariance on mutable non-`tensor` variables during the function runtime
- Forbidden direct acces to tensor buffers via `tensor::data()` or similar

Note that these restrictions can be sometimes desirable. For example,
we can use a condition or `while` cycle to _declare_ how we want our data 
to be processed, and these instructions will never have to be repeated again.

## Optimizations

Currently, JIT performs the following optimizations. For more in-depth 
description, refer to [`engine::Pass`](engine/lambda/passes).

- Constant propagation - pre-caches tensors that can be deterministically
  computed in compile-time rather than in run-time
- Copy propagation - eliminates unnecessary identities
- Dead code elimination (DCE) - prunes operations that no direct effect or
  side effect depends on
- Inline expansion - recursively inlines all nested Matcha functions to allow
  for further interprocedural optimizations
- Matmul fusion - fuses matrix multiplications with adjacent transpose operations

## To JIT or not to JIT

**JIT** when:

- The function contains **many operations** operating on **big data**
- The **overhead for operation initialization** is too large
- The function is somewhat **close to pure** and has only **`tensor` external parameters**

**Don't JIT** when:

- The function contains only **a few operations** 
- You want to **control the function** flow using many external parameters
- You want to directly **access tensor buffers**

