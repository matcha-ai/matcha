# Accessing tensor data
> `tensor::data() -> void*`

In practice, we wouldn't like the data to stay inside matcha all the time.
We may need to access it directly with our own software. For CPU-level 
access, this can be done very simply using the `tensor::data()` method.
This covers maybe 99% of all uses:

```cpp
tensor a = eye(3, 3) + 1;
auto data = (float*) a.data();

for (int i = 0; i < a.size(); i++) {
  std::cout << data[i] << " ";
}

std::cout << std::endl;
```

Output:

```txt
2 1 1 1 2 1 1 1 2 
```

!> When [tracing](tensor/tracing.md), direct access to tensor buffers 
   would be problematic for several reasons. Notably, Matcha 
   [`Lambda`](engine/lambda/) functions are represented as a 
   (directed, acyclical, multi-) graph composed from traced operations 
   and it would be very troublesome to impossible to trace what happens
   to the data.
   Therefore, access to tensor data inside traced scopes is **forbidden**. 
   If you want to use a specific operation that matcha does not provide
   out-of-the-box, subclass [`engine::Op`](engine/op/)
   (example [here](engine/op/README#example)).


## Optimized access

For the 1% of uses, when we want absolutely optimized access to tensor
data, maybe with our GPU or TPU, there are multiple other ways. However,
here we have to tap into the [`matcha::engine`](engine/).
To do this, you can derefernce the given [`tensor`](tensor/) into 
its internal [`engine::Tensor*`](engine/tensor/) using 
[`engine::deref`](engine/tensor/README#interface-binding). Now,
you can directly access the tensor's 
[`engine::Buffer`](engine/tensor/buffer) by calling the internal tensor's
[`engine::Tensor::buffer`](engine/tensor/README#buffer-methods) method.

