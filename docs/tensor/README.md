# Tensors
> `class tensor final`

Matcha provides a primitive-like type `tensor` to enable large-scale computations.
Tensors represent multidimensional arrays of data of a signle type (usually `Float`).


Creating a tensor can be as simple as:


```cpp
tensor a = 42;
```


We have just successfully created a tensor holding the scalar value 42.
But usually, we want to represent _much_ bigger data. For example, we may want to create
a 100x100 matrix with all values initialized to zero or to some custom values:

```cpp
tensor b = zeros(100, 100);
```

Tensors can be reassigned without a problem:

```cpp
tensor c = {{6, 8, 0, 2},
            {2, 0, 7, 1},
            {5, 2, 0, 9}};
c = uniform(100);
```

## Basic operations

Tensor arithmetic is a core functionality of Matcha.
It is designed to be done very intuitively, much like in Numpy:


```cpp
tensor d;           // forward-declaration is possible too
d = a + b;
d = a * (b + 2i);   // complex numbers
d = power(2, -b);   // and so on...
```

Matcha implements also linear algebra operations and various operations
for composing tensors:


```cpp
d = transpose(b);   // transposition
d = b.t();          // transposition, but shorter
d = matmul(b, c);   // matrix multiplication
d = b.cat(c);       // concatenating b and c into one tensor
```

You can learn more about specific operations e.g. in [the following article](tensor/operations/).

## Tensor frames

So far, we have created quite a diversity of tensors. To inspect their datatypes and shapes, we can use the methods:

```cpp
d = 1i * ones(50, 50);                // 50x50 matrix of complex units

std::cout << d.dtype() << std::endl;  // Cfloat
std::cout << d.shape() << std::endl;  // [50, 50]

```

In fact, the pair `Dtype, Shape` is actually the `Frame` of a tensor. Let's see what are the frames of the tensors
we have encountered this far:

```cpp
std::cout << a.frame() << std::endl;  // Int[]
std::cout << b.frame() << std::endl;  // Float[100, 100]
std::cout << c.frame() << std::endl;  // Float[100]
std::cout << d.frame() << std::endl;  // Cfloat[50, 50]
```

Typically, frames determine whether specific operations can or can not be performed and what the resulting frames will be:

```cpp
tensor e;

e = a.transpose();   // error: `a` is not vector or matrix-like
e = b.reshape(60);   // error: [100, 100] cannot be reshaped into [60]
e = b + d;           // error: trying to add together Float[100, 100] and Cfloat[50, 50]

e = a + b;           // OK! adding a scalar to any tensor is cool
e = exp(d);          // OK! elementwise exponential function does not care
```

Frames also determine the backend size required to store tensors. For example `c`, which is a 100-element array/vector,
will require 100-times the size of the underlying data type, which in this case is 4-byte `Float`. The total required backend size is therefore 400 bytes:

```cpp
std::cout << c.frame().bytes() << std::endl;  // 400
```

You may have been suspicious about this. Afterall, `sizeof(tensor)` always seems to return 8 bytes (or similar, this is platform-specific).
This is so, because `tensor` is in reality only an opaque pointer
to its internal [`engine::Tensor`](engine/tensor/) object. That internal object
holds the [`engine::Buffer`](engine/tensor/buffer) and other associated information,
like the tensor's [`Frame`](tensor/frames). Finally, the number of bytes 
is related only to the [`engine::Buffer`](engine/buffer) size. For details, read the articles
on engine [memory](engine/memory) and [`engine::Buffer`](engine/tensor/buffer).

For a more technical explanation of frames, read the [next article](tensor/frames).
