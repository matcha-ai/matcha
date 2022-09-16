# Reflection
> `"bits_of_matcha/engine/op/Reflection.h"`\
> `template <class Operation> struct engine::Reflection`

To enable the Matcha engine to perform some operation-level manipulations,
each operation must declare its `Reflection`. Reflection was designed to need
as little boilerplate code as possible, but still it has to be done.
Each `Reflection` instance should be static. This ensures it will be 
initialized right away and that it will be available to the Matcha engine
for the entire program runtime. When initialized, `Reflection` references
itself into the internal Matcha ops registry `engine::Registry`, 
which mediates all operation reflection within the engine. 

#### Public reflection members

- `std::string name` - operation type name
- `std::function<std::string(Operation*)> label` - operation instance label
- `std::function<Lambda(const BackCtx&)> back` - reverse mode derivative
- `bool deterministic = true` - whether an operation is deterministic (depends only on its inputs)
- `bool side_effect = false` - whether an operation has or does not have a side effect
- `std::function<Operation*(Operation*)> copy` - operation copy (runs the operation's copy constructor by default)

#### Internal logic

- `Reflection::RegisterCtx register_ctx_` - notifies the Matcha engine about new Reflections

!> The internal logic should not be modified directly. It is exposed
   to enable the C++20 [aggregate initialization](https://en.cppreference.com/w/cpp/language/aggregate_initialization)
   feature.

## Querying properties

- `engine::ops::name(Op*) -> std::string` - operation type name
- `engine::ops::label(Op*) -> std::string` - operation label
- `engine::ops::isSideEffect(Op*) -> bool` - side effect getter
- `engine::ops::isDeterministic(Op*) -> bool` - determinism getter
- `engine::ops::back(const BackCtx&) -> Lambda` - backpropagation builder
- `engine::ops::copy(Op*) -> Op*` - operation clone


## Example

Suppose we have inherited [`Op`](engine/op/) and created our own custom operation 
`MyOperation`. A simple reflection for `MyOperation` may look like this.

```cpp
// in the header file
struct MyOperation : Op {
  // ...
  static Reflection<MyOperation> reflection;
};
```

We will explicitly name the operation type `MyOperation`. 
For the sake of demonstration, we will also tell the Matcha engine it 
has some side effect. We can leave the other fields to their default
values.

```cpp
// in the source file
Reflection<MyOperation> MyOperation::reflection {
  .name = "MyOperation",
  .side_effect = true
};
```
Now, when we instantiate our `MyOperation`:

```cpp
Op* op = new MyOperation;
```

We can query its reflection as follows:


```cpp
std::cout << ops::name(op) << std::endl;
std::cout << ops::isDeterministic(op) << std::endl;
std::cout << ops::isSideEffect(op) << std::endl;
```

Output:

```txt
MyOperation
1
1
```

We can also clone it:

```cpp
Op* clone = ops::copy(op);
std::cout << ops::name(clone) << std::endl;
```

Output:

```txt
MyOperation
```

## Backpropagation example

Backpropagation is a complex functionality that is central to Matcha,
and deserves some extra attention.
