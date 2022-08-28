# Gotchas

## Overpromotion of 32-bit dtypes

For future compability with TPUs and GPUs, where `double` operations
are often unsupported, Matcha dtype promotion rules strive to preserve
bit size rather than representative capacity, which may, if not careful,
result in arithmetic overflows. For example, a binary operation on
`Int` and `Float` converts `Int` to `Float` rather than to `Double`, in
order to keep the kernel types 32-bit. The result of such operation is again
of type `Float`. For details, read 
[dtype promotion](tensor/operations/dtype-promotion).

## JIT native variable caching

When tracing, the dependence of **Matcha tensors** is inspected. This ignores
any **non-tensor** variables, such as global `bool` (even non-constant) 
variables controlling the flow inside the JITted function. When the function
is compiled and traced, these non-tensor variables play a role in determining
the function flow. After the compilation is complete, the compiled function
version is invoked instead of the original. Therefore, these non-tensor
variables no longer play a role, since their value cached during the
compilation is used.

The following code, for example, returns `43` three times, which
would be unexpected:

```cpp
int i = 0;

tensor foo(tensor x) {
  i += 1;
  return x + i;
}

int main() {
  auto joo = jit(foo);                // wrap foo by matcha JIT compiler
  std::cout << joo(42) << std::endl;  // foo is JIT compiled, `i` is increased to 1
  std::cout << joo(42) << std::endl;  // but here, the original function is not run again
  std::cout << joo(42) << std::endl;  // instead, the previously compiled matcha version is invoked
}
```

To fix that, simply declare `i` to be of type `tensor` instead:

```cpp
tensor i = 0;                         // `i` can be traced by matcha now

tensor foo(tensor x) {
  i += 1;
  return x + i;
}

int main() {
  auto joo = jit(foo);                // wrap foo by matcha JIT compiler
  std::cout << joo(42) << std::endl;  // foo is JIT compiled, including the `i` increment
  std::cout << joo(42) << std::endl;  // the compiled instructions are called again, including `i += 1`
  std::cout << joo(42) << std::endl;  // and once more
}
```

Output:

```text
43
44
45
```

## Neural network dynamism

In contrast to static machine learning frameworks like 
[TensorFlow](https://www.tensorflow.org/), which let you merely
_declare_ the network topology through enumerating its components,
Matcha allows more flexibility through dynamic flow. However, this
means that all components that the neural net uses must be stored
somewhere so that they can be explicitly called later in your code.
For example, instantiating and calling a layer all at once inside
the `run` method **would not work as expected in TensorFlow**. Instead,
a new layer would be created in each `run` invokation. Therefore,
**you must instantiate all layers before calling `run`**, e.g. from `init`
as private class members.

The following following functional API code **is incorrect**.
New `nn::Fc` layers will be instantiated in every forward propagation:

```cpp
Net net = (fn) [](tensor x) {
  x = nn::Fc{100, "relu"}(x);
  x = nn::Fc{1, "exp"}(x);
  return x;
};
```

Instead, each stateful component must be made persistent in some way:

```cpp
Net net = (fn) [](tensor x) {
  static auto hidden = nn::Fc{100, "relu"};
  static auto output = nn::Fc{1, "exp"};

  x = output(hidden(x));
  return x;
};
```

Alternatively, although not so elegant, using reference-capturing lambdas:

```cpp
auto hidden = nn::Fc{100, "relu"};
auto output = nn::Fc{1, "exp"};

Net net = (fn) [&](tensor x) {
  x = output(hidden(x));
  return x;
};
```

Or using the subclassing API:

```cpp
class MyNet : public Net {
  auto hidden = nn::Fc{100, "relu"};
  auto output = nn::Fc{1, "exp"};

  tensor run(const tensor& x) override {
    return output(hidden(x));
  }
};
```
