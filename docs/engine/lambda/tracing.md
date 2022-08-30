# Tracing

> `engine::trace(const fn& function, const std::vector<Frame>& frames) -> Lambda`

Tracing is a technique for inspecting what happens within some block of code.
The Matcha Tracing system is able to trace tensor operations in user-defined
functions in the program runtime. 
The result of a tracing process is a valid `Lambda`. For details 
and limitations, refer to [this article](tensor/tracing).

## Tracer

> `engine::Tracer`

Class faciliating potentially recursive tracing.

#### Control methods

- `open(const std::vector<Frame>& frames) -> std::vector<tensor>` - initializes the tracing process and returns inputs
- `close(const std::vector<tensor>& outputs) -> Lambda` - finalizes the tracing process and returns the resulting lambda

#### Tracer lifecycle

- a `Tracer` is instantiated
- its `open(const std::vector<Frame>& frames)` method is called, returning
  traced input tensors
- tensors and dispatched operations are handed to the Tracer, instead of being run and cleaned up as usual
- the tracer's `close(const std::vector<tensor>& outputs)` method is called
  on the output tensors, finalizes the tracing process and returns the resulting `Lambda`
- the tracer is destroyed

