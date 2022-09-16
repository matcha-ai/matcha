# Engine backend

For practical reasons, the internal Matcha `engine` layer is altogether
separated from Matcha's interface. This documentation section
focuses on the internal engine layer. In the interface, most
technical things are abstracted away. Here one deals with them.
Familiarity with general Matcha concepts (e.g. [`tensor`](tensor/), 
[`Frame`](tensor/frames)) is assumed.


## Brief overview

- [`engine::Buffer`](engine/tensor/buffer) - a wrapper for a contiguous 
  block of memory on some device (e.g. RAM)
- [`engine::Tensor`](engine/tensor/) - the [`tensor`](tensor/) backend object, 
   essentially [`engine::Buffer`](engine/tensor/buffer) and 
  [`Frame`](tensor/frames), with some context around
- [`engine::Op`](engine/op/) - base operation class, 
  all operations inherit from it; has tensor `inputs` and `outputs`
- [`engine::Lambda`](engine/lambda/) - a sequence of interconnected 
  [`engine::Op`](engine/op/) operation objects on tensors, 
  also has its tensor `inputs` and `outputs`
- [`engine::Transform`](engine/transform/) - base class for functional
  transformations just-in-time


## Philosophy

- Keep the `engine` namespace altogether hidden from the outer interface. 
  This often means [PIMPLing](https://en.cppreference.com/w/cpp/language/pimpl) everything.
- Usually, time is more important than memory.
- Unless there is a reason not to, use the STL. This includes smart pointers.
- Use existing ecosystem of [kernels](./kernels/), unless inefficient.
- Make sure your operations work with both eager and lazy execution.
  This implies declaring its [`Reflection`](engine/op/reflection).
