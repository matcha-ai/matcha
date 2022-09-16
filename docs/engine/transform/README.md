# Transforms

Use transforms to decorate the preimage function just-in-time (JIT).


## Base transform
> `"bits_of_matcha/engine/transform/Transform.h"`\
> `class engine::Transform`

Base tensor function transform.

#### Constructors

- `explicit Transform(const fn& preimage)` - wraps the preimage function
- `explicit Transform()` - uninitialized preimage function

#### Public virtual methods
- `virtual run(const std::vector<Tensor*>&) -> std::vector<Tensor*>` - function call

#### Other public methods

- `preimge() -> fn&` - preimage getter
- `preimge() const -> const fn&` - const preimage getter
- `hasPreimage() bool` - preimage presence getter
- `setPreimage(const fn&) -> void` - preimage setter

## CachingTransform
> `"bits_of_matcha/engine/transform/CachingTransform.h"`\
> `class engine::CachingTransform : public engine::Transform`

When `run` is called, `CachingTransform` [traces](engine/lambda/tracing)
the preimage function and compiles it with overridable logic.
Any subsequent `run` invokations use previously compiled instructions whenever
available, based on the lambda input Frames.

#### Protected virtual methods

- `virtual compile(Lambda) -> std::shared_ptr<engine::Executor>` -
  compilation logic

#### Other public methods

- `build(const std::vector<Tensor*>&)` - 
  compiles the preimage function for given inputs
- `build(const std::vector<Frame>&)` - 
  compiles the preimage function for given inputs
- `cache(const std::vector<Tensro*>&) -> std::shared_ptr<engine::Executor>`-
  retrieves cached instructions for given inputs (builds if not cached yet)
- `cache(const std::vector<Frame>&) -> std::shared_ptr<engine::Executor>`-
  retrieves cached instructions for given inputs (builds if not cached yet)

## JitTransform
> `"bits_of_matcha/engine/transform/JitTransform.h"`\
> `class engine::JitTransform : public CachingTransform`

Internal transform wrapper for [`jit`](tensor/jit). 
For supported optimizations, see [this section](tensor/jit#optimizations).

## Example

We will create inherit 
[`engine::Transform`](engine/transform/README#base-transform)
to create our custom `MyDebugTransform`. In each invocation, it will
trace the preimage function on the current inputs, debug it, and run it.
We will want this to be done on per-call basis, as opposed to
lambda-caching. For that, we would inherit from
[`engine::CachingTransform`](engine/transform/README#cachingtransform)
instead:

```cpp
class MyDebugTransform : public Transform {
  MyDebugTransform(const fn& function) : Transform(function) {}

  std::vector<Tensor*> run(const std::vector<Tensor*>& inputs) override;
};
```

Implementing the logic is qute simple. Take the input frames, pass
them to the Matcha tracer together with the preimage function.
Then debug the lambda, and finally init + run it:


```cpp
std::vector<Tensor*> MyDebugTransform::run(const std::vector<Tensor*>& inputs) {
  std::vector<Frame> frames;
  for (auto&& input: inputs)
    frames.push_back(input->frame());

  Lambda lambda = trace(preimage(), frames);

  debug(lambda);

  init(lambda);
  SinglecoreExecutor executor(std::move(lambda));
  return executor.run(inputs);
}
```

Done. Proceed to [transformation binding](engine/transform/binding) to
learn how to integrate this into the rest of Matcha.
