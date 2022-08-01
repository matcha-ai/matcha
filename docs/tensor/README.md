# Tensors

Matcha provides a primitive-like object `tensor` to enable large-scale computations.
Tensors represent multidimensional arrays of data of a signle type (usually `Float`).


Creating a tensor can be as simple as:


```cpp
tensor a = 42;
```


We have just successfully created a tensor holding the scalar value 42.
But usually, we want to represent _much_ bigger data. For example, we may want to create
a 100x100 matrix with all values initialized to zero:

```cpp
tensor b = tensor::zeros(100, 100);
```

Tensors can be reassigned:

```cpp
tensor c = 69;
c = tensor::zeros(100);
```

## Operator methods

The `tensor` object provides shortcut methods for some tensor operations.
You can learn more about those e.g. in [the following article](tensor/basic-arithmetic).

Notably, `tensor` overloads arithmetic operators for readable and natural code such as:

```cpp
tensor d;    // forward_-declaration is possible too
d = a + b;
d = a * b;
d = -b;

// and so on...
```

Unluckily, there is only few overloadable operators; `tensor` provides helper methods even
for operations that don't have any overloadable operator syntax, such as:

```cpp
d = b.pow(a);       // a-th power
d = b.transpose();  // transposition
d = b.t();          // transposition, but shorter
d = b.dot(c);       // the dot product
d = b.cat(b);       // concatenating b and b into one tensor
```

## Tensor frames

So far, we have created quite a diversity of tensors. To inspect their datatypes and shapes, we can use the methods:

```cpp
d = tensor::ones(50, 50);

std::cout << d.dtype() << std::endl;  // Float
std::cout << d.shape() << std::endl;  // [50, 50]

```

In fact, the tuple `Dtype` and `Shape` is actually the `Frame` of a tensor. Let's see what are the frames of the tensors
we have encountered this far:

```cpp
std::cout << a.frame() << std::endl;  // Float[]
std::cout << b.frame() << std::endl;  // Float[100, 100]
std::cout << c.frame() << std::endl;  // Float[100]
std::cout << d.frame() << std::endl;  // Float[50, 50]
```

Usually, frames determine whether specific operations can or can not be performed on given tensors:

```cpp
tensor e;

e = a.transpose();   // error: `a` is not vector or matrix-like
e = b.reshape(60);   // error: [100, 100] cannot be reshaped into [60]
e = b + d;           // error: trying to add together Float[100, 100] and Float[50, 50]

e = a + b;           // OK! adding a scalar to any tensor is cool
e = b.dot(c);        // OK! dot product between a matrix and an appropriate vector is cool too
e = exp(d);          // OK! elementwise exponential function does not care
```

Frames also determine the backend size required to store tensors. For example `c`, which is a 100-element array/vector,
will require 100-times the size of the underlying data type, which in this case is 4-byte `Float`. The total required backend size is therefore 400 bytes:

```cpp
std::cout << c.frame().bytes() << std::endl;  // 400
```

You may have been suspicious about this. Afterall, `sizeof(tensor)` always seems to return 8 bytes (or similar, this is platform-specific).
This is so because `tensor` is in reality an implicit reference to some [internal engine tensor object](engine/tensor). That engine object
holds the `Buffer` and associated metadata, like the `Frame`. Finally, the number of bytes is related only to the `Buffer` size. For details, read the articles
on engine [Memory](engine/memory) and [Buffers](engine/buffers).
