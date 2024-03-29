# Benchmarks

This section deals with benchmarking Matcha linked with 
[OpenBLAS](https://github.com/xianyi/OpenBLAS), and
using the following optimization flags:

```sh
-march=native
-O3
```

Hardware used for benchmarks:

_Dell XPS 13 9370 - x86\_64 architecture Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz_


!> When benchmarking against accelerated libraries, such as 
   [TensorFlow](https://www.tensorflow.org/), 
   the same  devices are always used. Since Matcha does not yet support 
   GPUs or TPUs, these libraries will be forced to use the CPU.
   Unaccelerated libraries, like [Numpy](https://numpy.org/), are unaffected.

!> As for example Numpy is a Python library, there
   is some overhead associated with the Python language itself.
   This however turns out to be negligible, as most of the
   actual weightlifting is done in C anyway.

For details about benchmark generation, see 
[this GitHub repo](https://github.com/matcha-ai/benchmark).


### Available benchmarking data:

- [Operations](benchmarks/ops) - performance of tensor operations
- [JIT compilation](benchmarks/jit) - effects of JIT optimizations and overheads of JITing
- [Automatic differentiation](benchmarks/autograd) - automatic differentiation performance
- [Neural networks](benchmarks/nn) - performance of training and generating predictions
