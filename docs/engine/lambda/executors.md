# Executors

When executing eagerly, an operation is constructed, initialized, run,
and destroyed, all at once. In lazy scheduling, a `Lambda` is passed to
an `Executor`, an object that handles the potentially repeated execution
of operations in the lambda.

## Base executor

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
> `class engine::SinglecoreExecutor`

Executes its lambda in the same thread it is called from. 
When initialized, `SinglecoreExecutor` analyzes the lambda's 
tensor dependencies and prepares an aggressive memory freeing policy
for the runtime.
