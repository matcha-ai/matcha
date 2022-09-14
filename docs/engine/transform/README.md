# Transforms

Use transforms to decorate the preimage function just-in-time (JIT).


## Base transform
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
> `class engine::JitTransform : public CachingTransform`

JIT compilation transform. For supported optimizations, see
[this article](tensor/jit#optimizations).

