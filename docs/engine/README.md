# Engine backend

For practical reasons, the internal Matcha `engine` layer is altogether
separated from Matcha's interface. This documentation section
focuses on the internal engine layer. In the interface, most
technical things are abstracted away. Here one deals with them.
Familiarity with general Matcha concepts (e.g. `tensor`, `Frame`)
is assumed.


## Brief overview

- `engine::Buffer` - a wrapper for a contiguous block of memory on some device (e.g. RAM)
- `engine::Tensor` - the `tensor` backend object, essentially `engine::Buffer` and `Frame`, with some context around
- `engine::Op` - base operation class, all operations inherit from it; has `inputs` and `outputs`
- `engine::Lambda` - a sequence of interconnected `engine::Op` operation objects on tensors, also has its inputs and outputs
- `engine::Transform` - base class for modifying and replacing normal functions by their enhanced counterpart


## Philosophy

- Keep the `engine` namespace altogether hidden from the outer interface. 
  This often means [PIMPLing](https://en.cppreference.com/w/cpp/language/pimpl) everything.
- Usually, time is more important than memory.
- Unless there is a reason not to, use the STL. This includes smart pointers.
- Use existing ecosystem of [kernels](./kernels/), unless inefficient.
- Make sure your operations work with both eager and lazy execution. This implies including [Reflection](engine/op/reflection).
