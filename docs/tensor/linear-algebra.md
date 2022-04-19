# Linear algebra

## transpose
 
> `transpose(const tensor& a)` \
> `tensor::transpose()`, `tensor::t()`

Transpose vector or matrix-like tensor. Note that you can use `tensor::transpose` or `tensor::t` instead:

```cpp
tensor a = tensor::ones(100);
tensor b = tensor::ones(100, 3);

tensor c = transpose(a);  // => Float[100, 1]
tensor d = transpose(b);  // => Float[3, 100]
```

## dot

> `dot(const tensor& a, const tensor& b)` \
> `tensor::dot(const tensor& b)`

Dot product of two vector or matrix-like tensors. Note that you can use `tensor::dot` instead:

```cpp
tensor a = tensor::ones(10, 20);
tensor b = tensor::ones(20, 30);

tensor c = dot(a, b);  // => Float[10, 30]
```
