# Benchmarks

This section deals with benchmarking Matcha linked with 
[OpenBLAS](https://github.com/xianyi/OpenBLAS), and
using the following optimization flags:

```sh
-march=native
-O3
```

Hardware used for benchmarks:

_Dell XPS 13 9370, x86\_64 architecture Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz_


!> When benchmarking against accelerated libraries, such as 
   [TensorFlow](https://www.tensorflow.org/), 
   the same  devices are always used. Since Matcha does not yet support 
   GPUs or TPUs, these libraries will be forced to use the CPU.
   Unaccelerated libraries, like [Numpy](https://numpy.org/), are unaffected.
