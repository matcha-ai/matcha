# Constant propagation
> `engine::constantPropagation(Lambda&) -> void`

Passes through the lambda, runs all _deterministic_ non-_side-effect_
operations depending on constant tensors only, and prunes them.

## Example

Consider the following function:

```cpp
tensor foo(const tensor& x) {
  tensor temp = 3;
  temp += 4;
  return x * temp;
}
```

Lambda **before the pass**:

```txt
lambda(a: Float[3, 3]) -> Float[3, 3] {
    d = Add(b, c)
    e = Cast(d)
    f = Multiply(a, e)
    g = Identity(f)

    return g
}
```

Here, `d` is the result of `3 + 4`, which is then casted to Float `e`,
in order to be multiplied with Float input. The addition and cast can
be statically inferred, since it is deterministic and depends on
constants only.

Lambda **after the pass**:

```txt
lambda(a: Float[3, 3]) -> Float[3, 3] {
    c = Multiply(a, b)
    d = Identity(c)

    return d
}
```

The addition and cast has been pre-computed.

## Op implementation requirements

Constant propagation queries operations on the following 
[reflection](engine/op/reflection) properties:

- `bool side_effect`
- `bool deterministic`

