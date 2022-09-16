# JIT limitations and gotchas

## Forbidden access to tensor data

When [tracing](tensor/tracing.md), direct access to tensor buffers 
would be problematic for several reasons. Notably, Matcha 
[`engine::Lambda`](engine/lambda/) is represented as a 
(directed, acyclical, multi-) graph composed from traced operations 
and it would be very troublesome to impossible to trace what happens
to the data.
Therefore, access to tensor data in traced code is **forbidden**. 
  If you want to use a specific operation that matcha does not provide
out-of-the-box, subclass [`engine::Op`](engine/op/)
  (example [here](engine/op/example)).

Example function:

```cpp
tensor foo(tensor a) {
  auto data = (float*) a.data();

  for (size_t i = 0; i < a.size(); i++)
    data[i] *= 2;

  return a;
}
```

When JITing the function and calling it, `std::runtime_error` is thrown:

```cpp
int main() {
  auto joo = jit(foo);

  tensor b = joo(ones(10);
  std::cout << b << std::endl;
};
```

Output:

```txt
terminate called after calling an instance of `std::runtime_error`
  what():  reading tensor data in traced code is forbidden
```


## Inhibited IO side effects

Suppose a function that has some IO side effects, in this case, printing:

```cpp
tensor foo(tensor a) {
  tensor b = sum(a);
  std::cout << "calculated sum: " << b << std::endl;
  return b;
}
```

Consider JITing the function and running it several times:

```cpp
int main() {
  auto joo = jit(foo);
  tensor a = {1, 2.5, 3, 4};

  joo(a);
  joo(a);
  joo(a);
}
```

Contrarily to what we would expect, the snippet produces this output:

```txt
calculated sum: Float[]
```

We probably expected three lines reporting the actual sum (`10.5`).
Instead, the line is only one and the result is not even there. This happens
because the function that we defined (with the `std::cout` line) was
run only once - and in tracing mode. 

While tracing, nothing was actually
computed. Only tensor frames were inferred and tensor relationships were
captured. Therefore, the only known thing at that time
to print to the standard output was the tensor frame - `Float[]`.

Since tracing can be done only on
Matcha-provided tensors and operations, the traced
function version does not contain the printing at all.
Therefore, the subsequent
calls to that already cached version could not invoke any.

This can be verified by calling `joo` again, this time on a new frame.
When JIT encounters inputs with unfamiliar frames, the whole process
is repeated and a new optimized function is cached. Let's change the
snippet from above:

```cpp
int main() {
  auto joo = jit(foo);
  tensor a = {1, 2.5, 3, 4};

  joo(a);
  joo(a.cast(Double);
  joo(stack(a, a));
}
```

Output:

```txt
calculated sum: Float[]
calculated sum: Double[]
calculated sum: Float[]
```

This time, we first invoked `joo` on `Float[4]`. Then we called it on
`Double[4]`, and the last time, it was run on `Float[2, 4]`. Since
each of these inputs is of a different frame, no appropriate lambda
could be found in the JIT cache, and every call caused re-tracing.
Note that again, only frames were inferred and not the actual value.

Of course, this limitation applies for traced code only. We can naturally
read the actual result of the function outside of JIT:

```cpp
int main() {
  auto joo = jit(foo);
  tensor a = {1, 2.5, 3, 4};

  tensor result = joo(a);
  std::cout << "the actual result is: " << result << std::endl;
}
```

Output:

```txt
calculated sum: Float[]
the actual result is: 10.5
```


## Static flow control

Consider the following function:

```cpp
tensor foo(tensor a) {
  if (all(a > 0))
    return 8;
  else
    return 1;
}
```

JITting it produces would produce an error. Reliably tracing through
flow-controlling structures is not possible in C++. Instead, work-arounds
have to be made. In this case like this:

```cpp
tensor foo(tensor a) {
  tensor cond = a > 0;
  return cond * 7 + 1;
}
```

Matcha-compatible differnetiable flow control statements are planned 
in future versions. Also, note that flow control based on non-tensor
variables is possible, but is subject to caching. See below.

Note that these restrictions can be sometimes desirable. For example,
we can use a condition or `while` cycle to _declare_ how we want our data 
to be processed, and these instructions will never have to be repeated again.


## Caching native variables

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


