# Executors

When executing eagerly, an operation is constructed, initialized, run,
and destroyed, all at once. In lazy scheduling, an [`engine::Lambda`](engine/lambda/) is passed to
an [`engine::Executor`](engine/lambda/executors), an object that handles the potentially repeated execution
of operations in the lambda.

## Base executor

> `"bits_of_matcha/engine/lambda/Executor.h"`\
> `class engine::Executor` - abstract base executor class

#### Constructors

- `explicit Executor(Lambda&& lambda)` - construct an executor for given lambda

#### Virtual methods

- `runInternal() -> void = 0` - executes the lambda, the lambda inputs are assumed to 
  hold the required data, at the end, outputs are assumed to hold the resulting data
- `run(const std::vector<Tensor*>& ins, const std::vector<Tensor*>& outs) -> void` - executes 
  the lambda on input data provied in `ins` and streams the outputs into `outs`
  - calls `runInternal()` by default
- `run(const std::vector<Tensor*>& ins) -> std::vector<Tensor*>` - executes 
  the lambda on input data provied in `ins`, creates new output tensors, and stream the results there
  - calls `run(ins, outs)` by default

#### Getters

- `lambda() -> Lambda&` - retrieves the executor's lambda
- `lambda() const -> const Lambda&` - retrieves the executor's lambda


## Implementations

#### SinglecoreExecutor
> `"bits_of_matcha/engine/lambda/executors/SinglecoreExecutor.h"`\
> `class engine::SinglecoreExecutor`

Executes its lambda in the same thread it is called from. 
When initialized, `SinglecoreExecutor` analyzes the lambda's 
tensor dependencies and prepares a policy for freeing tensor
[`engine::Buffer`](engine/tensor/buffer) data as soon as it is no longer needed.

## Examples

Suppose we are given a valid [`engine::Lambda`](engine/lambda/) and want to run it:

```cpp
using namespace matcha::engine;

Lambda lambda /* = ... */;
```

First, we initialize it, if it has not yet been done:

```cpp
init(lambda);
```

Then we instantiate the `SinglecoreExecutor`. Since executors take
ownership of their lambda, and we do not want to create a deep copy of it,
we will use the [C++ move semantics](https://en.cppreference.com/w/cpp/utility/move).
Note that moving the lambda into the executor this will make the original
`lambda` variable empty:

```cpp
SinglecoreExecutor executor(std::move(lambda));
```

Now, we can run `executor` repeatedly, specifying our inputs, and
collecting its outputs. Note that all the inputs are assumed to be
of the right [`Frame`](tensor/frames).

```cpp
std::vector<Tensor*> inputs /* = ... */;
std::vector<Tensor*> outputs = executor.run(inputs);
```

At the end, the executor's destructor automatically deallocates all 
lambda resources.
