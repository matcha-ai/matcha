# Accessing tensor data

In practice, we wouldn't like the data to stay inside matcha all the time. We may need to access it directly with our own software.
For CPU-level access, this can be done very simply using the `tensor::data()` method. This covers maybe 99% of all uses:

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

!> Inside the Flow, access to tensors would be problematic for several reasons.
Notably, Flow is represented as a Graph composed from traced Operations and
it would be very troublesome to impossible to trace what happens to the data.
Therefore, access to tensor data inside the Flow is **forbidden**. If you want to
use a specific operation that matcha does not provide out-of-the-box, you
must first register it as a [custom Operation](custom-op.md).


## Optimized access

For the 1% of uses, when we want absolutely optimized access to tensor
data, maybe with our GPU or TPU, there are multiple other ways. However,
here we have to tap into the [engine layer](engine) of Matcha.
