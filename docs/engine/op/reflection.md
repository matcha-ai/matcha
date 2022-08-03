# Reflection

> `template <class Operation> engine::Reflection`

To enable the Matcha engine to perform some operation-level manipulations,
each operation must declare its `Reflection`. Reflection was designed to need
as little boilerplate code as possible, but still it has to be done.
Each `Reflection` instance should be static. This ensures it will be 
initialized right away and that it will be available to the Matcha engine
for the entire program runtime. When initialized, `Reflection` references
itself into the internal Matcha ops registry `engine::Registry`, 
which mediates all operation reflection within the engine. Let's suppose
we have inherited `Op` and created our own custom operation `CustomOp`.
A simple refleciton for `CustomOp` may look like this.

```h
// in the header file
struct CustomOp : Op {
  // ...
  static Reflection<CustomOp> reflection;
};
```

```cpp
// in the source file
Reflection<CustomOp> CustomOp::reflection {
  .name = "CustomOp",
};
```

## Public reflection members

- `std::string name` - operation type name
- `std::function<std::string(Operation*)> label` - operation instance label
- `std::function<Chain(const BackCtx&)> back` - reverse mode derivative
- `std::function<bool(Operation*)> side_effect` - whether an operation has or does not have a side effect (default false)
- `std::function<Operation*(Operation*)> copy` - operation copy (runs the operation's copy constructor by default)

## Other public members

- `Reflection::RegisterCtx register_ctx_` - notifies the Matcha engine about new Reflections


