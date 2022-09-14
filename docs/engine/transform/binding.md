# Transform binding
> `engine::ref(std::shared_ptr<Transform> transform) -> fn`

To seamlessly bind custom function transforms derived from 
[`engine::Transform`](engine/transform/) into the interface,
use `engine::ref`. The function explicitly accepts a standard shared
pointer `std::shared_ptr` to guarantee memory safety. It returns
a polymorphic tensor function type `fn`.

## Example

First, suppose we have inherited [`engine::Transform`](engine/transform)
and created our
own function transformation. We will call this class `MyDebugTransform`.
In each invocation, it will trace the function, debug it, and run it.
We will want this to be done on per-call basis, as opposed to
lambda-caching. For that, we would inherit from
[`engine::CachingTransform`](engine/transform/README#cachingtransform)
instead.

```cpp
class MyDebugTransform : public engine::Transform {
  MyDebugTransform(const fn& function) : Transform(function) {}

  std::vector<Tensor*> run(const std::vector<Tensor*>& inputs) override;
}
```

Implementing the logic is simple. Take the input frames, pass
them to the Matcha tracer together with the preimage function.
Then debug the lambda, and finally init + run it.

```cpp
std::vector<Tensor*> MyDebugTransform::run(const std::vector<Tensor*>& inputs) {
  using namespace matcha::engine;

  std::vector<Frame> frames;
  for (auto&& input: inputs)
    frames.push_back(input->frame());

  Lambda lambda = trace(preimage(), frames);

  debug(lambda);

  init(lambda);
  SinglecoreExecutor executor(std::move(lambda));
  return executor.run(inputs);
}
```

Now, it is time to create the binding logic for our transform into the interface:

```cpp
fn myDebug(const fn& function) {
  std::shared_ptr<engine::Transform> internal { new MyDebugTransform(function) };
  return engine::ref(internal);
}
```

Done. We can now transform functions and call them:

```cpp
int main() {
  auto foo = myDebug(matcha::tanh);

  std::cout << foo(ones(3, 3)) << std::endl;
}
```

Output:

```txt
============================== LAMBDA DEBUG BEGIN ==============================

lambda(a: Float[3, 3]) -> Float[3, 3] {
    c = Cast(b)
    d = Multiply(c, a)
    e = Negative(d)
    f = Exp(e)
    h = Cast(g)
    i = Add(h, f)
    k = Cast(j)
    l = Divide(k, i)
    n = Cast(m)
    o = Multiply(n, l)
    p = Identity(o)

    return p
}

a 	op: 0
b 	op: 0
c 	op: 0x55a09c251630 (Cast)
d 	op: 0x55a09c251460 (Multiply)
e 	op: 0x55a09c2517f0 (Negative)
f 	op: 0x55a09c2519e0 (Exp)
g 	op: 0
h 	op: 0x55a09c251e20 (Cast)
i 	op: 0x55a09c251c30 (Add)
j 	op: 0
k 	op: 0x55a09c2522f0 (Cast)
l 	op: 0x55a09c252100 (Divide)
m 	op: 0
n 	op: 0x55a09c251bf0 (Cast)
o 	op: 0x55a09c2524f0 (Multiply)
p 	op: 0x55a09c2517b0 (Identity)

=============================== LAMBDA DEBUG END ===============================
[[1.76159 1.76159 1.76159]
 [1.76159 1.76159 1.76159]
 [1.76159 1.76159 1.76159]]
```
