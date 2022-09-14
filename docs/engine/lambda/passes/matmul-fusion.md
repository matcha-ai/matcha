# Matmul fusion
> `engine::matmulFusion(Lambda&) -> void`

Passes through the lambda and fuses all matrix multiplications 
with adjacent transpose operations into a single operation.

?> This leverages the underlying BLAS kernel option to iterate through
   a matrix in a transposed manner while performing the dot-product, 
   instead of creating a transposed copy
   before running the dot product itself. Note that the fused matrix
   multiplication operation is fully differentiable.

## Example 

Consider the following function:

```cpp
tensor foo(const tensor& x) {
  return exp(matmul(x, x.t()));
}
```

Lambda **before the pass**:

```txt
lambda(a: Float[3, 3]) -> Float[3, 3] {
    b = Transpose(a)
    c = Matmul(a, b)
    d = Exp(c)
    e = Identity(d)

    return e
}
```

**After the pass**, the transposition has been fused into 
the matrix multiplication:

```txt
lambda(a: Float[3, 3]) -> Float[3, 3] {
    b = Matmul(a, a)
    c = Identity(b)
    d = Exp(c)
    e = Identity(d)

    return e
}
```

!> Note that the lambda representation captures only the operation _types_
   and not additional options, such as the transpostition mask. \
   Therefore, the fused matrix multiplication is
   still represented just as `Matmul`.

## Op implementation requirements

Matmul fusion not query operations on any
[reflection](engine/op/reflection) property.
