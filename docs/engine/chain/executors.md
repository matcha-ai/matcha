# Executors

When executing eagerly, an operation is constructed, initialized, run,
and destroyed, all at once. In lazy scheduling, a `Chain` is passed to
an `Executor`, an object that handles the potentially repeated execution
of operations in the chain:

## Base executor

> `engine::Executor` - abstract base executor class

### Constructors

- `explicit Executor(Chain&& chain)` - construct an executor for given chain

### Virtual methods

- `runInternal() -> void = 0` - executes the chain, the chain inputs are assumed to 
  hold the required data, at the end, outputs are assumed to hold the resulting data
- `run(const std::vector<Tensor*>& ins, const std::vector<Tensor*>& outs) -> void` - executes 
  the chain on input data provied in `ins` and streams the outputs into `outs`
  - calls `runInternal()` by default
- `run(const std::vector<Tensor*>& ins) -> std::vector<Tensor*>` - executes 
  the chain on input data provied in `ins`, creates new output tensors, and stream the results there
  - calls `run(ins, outs)` by default

### Getters

- `chain() -> Chain&` - retrieves the executor's chain
- `chain() const -> const Chain&` - retrieves the executor's chain


## Implementations

- `engine::SinglecoreExecutor` - executes the chain in the same thread it is called from

