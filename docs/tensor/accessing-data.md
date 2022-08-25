# Accessing tensor data

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
   [functions](engine/lambda/) are represented as a 
   (directed, acyclical, multi-) graph composed from traced operations 
   and it would be very troublesome to impossible to trace what happens
   to the data.
   Therefore, access to tensor data inside traced scopes is **forbidden**. 
   If you want to use a specific operation that matcha does not provide
   out-of-the-box, subclass [`engine::Op`](engine/op/)
   (example [here](engine/op/example)).


## Optimized access

For the 1% of uses, when we want absolutely optimized access to tensor
data, maybe with our GPU or TPU, there are multiple other ways. However,
here we have to tap into the [`matcha::engine`](engine/).
