# Gotchas

## JIT native variable caching

When tracing, the dependence of **Matcha tensors** is inspected. This ignores
any **non-tensor** variables, such as global `bool` (even non-constant) 
variables controlling the flow inside the JITted function. When the function
is compiled and traced, these non-tensor variables play a role in determining
the function flow. After the compilation is complete, the compiled function
version is invoked instead of the original. Therefore, these non-tensor
variables no longer play a role, since their value cached during the
compilation is used.

The following code, for example, returns `42` three times, which
would be unexpected:

```cpp
bool change_foo = false;

tensor foo(tensor x) {
  if (!change_foo) {
    change_foo = true;    // in the first call,
    return x;             // return x
  } else {
    return x - 1;         // in all subsequent calls, return x - 1
  }
}

int main() {
  auto joo = jit(foo);                // wrap foo by matcha JIT compiler
  std::cout << joo(42) << std::endl;  // foo is JIT compiled, change_foo is set to true in the process
  std::cout << joo(42) << std::endl;  // but it is irrelevant here, because the compiled flow is used
  std::cout << joo(42) << std::endl;  // again, the cached flow from the compilation is used
}

```

## Neural network dynamism

In contrast to static machine learning frameworks like 
[TensorFlow](https://www.tensorflow.org/), which let you merely
_declare_ the network topology through enumerating its components,
Matcha allows more flexibility through dynamic flow. However, this
means that all components that the neural net uses must be stored
somethere so that they can be explicitly called later in your code.
For example, instantiating and calling a layer all at once inside
the `run` method **would not work as expected in TensorFlow**. Instead,
a new layer would be created in each `run` invokation. Therefore,
**you must instantiate all layers before calling `run`**, e.g. from `init`
as private class members.

The following following functional API code **is incorrect**.
A new `nn::Fc` layers will be instantiated in every forward propagation:

```cpp
Net net = (fn) [](tensor x) {
  x = nn::Fc{100, "relu"}(x);
  x = nn::Fc{1, "exp"}(x);
  return x;
};
```

Instead, each stateful object must be made persistent in some way:

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
