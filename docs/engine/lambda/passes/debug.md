# Debug
> `"bits_of_matcha/engine/lambda/passes/debug.h"`\
> `engine::debug(const Lambda&, std::ostream& os = std::cerr) -> void`

Prints the lambda and related debugging info, runs [`engine::check`](engine/lambda/passes/check).
Output stream can be specified. 
Be default, the standard error stream `std::cerr` is used.

## Example

Suppose the following function:

```cpp
tensor i = 0;

tensor foo(const tensor& x) {
  i += 1;
  return exp(matmul(x, x.t()));
}
```


Debug output (without previous passes, such as optimizations):

```txt
============================== LAMBDA DEBUG BEGIN ==============================

lambda(a: Float[3, 3]) -> Float[3, 3] {
    d = Add(b, c)
    e = Transpose(a)
    f = Matmul(a, e)
    g = Exp(f)
    h = Identity(g)
    SideOutput(d)

    return h
}

a 	op: 0
c 	op: 0
b 	op: 0	 side in: 0x5616a3cb7160 (Int[])
d 	op: 0x5616a4970250 (Add)
e 	op: 0x5616a4970530 (Transpose)
f 	op: 0x5616a49706e0 (Matmul)
g 	op: 0x5616a4970940 (Exp)
h 	op: 0x5616a4970900 (Identity)

=============================== LAMBDA DEBUG END ===============================
```

## Output exaplanation

- First, **lambda representation** is shown. 
  This follows the standard lambda printing format.
- Then **individual tensor information**:
  - Source operation - memory location, 
    type (if declared in the operation's [`Reflection`](engine/op/reflection))
  - Its side input outside of the lambda - memory location, frame
- If the lambda isn't valid, a **warning** is shown at the end.

## Op implementation requirements

Debug does not require operations to have their
[`Reflection`](engine/op/reflection) declared. However, it optionally queries
the following properties:

- `std::string name`
