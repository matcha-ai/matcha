#include "bits_of_matcha/engine/ops/Sum.h"

#include <algorithm>
#include <numeric>
#include <execution>


namespace matcha::engine::ops {

Sum::Sum(Tensor* a)
  : AxiswiseFoldOp(a)
{}

Sum::Sum(Tensor* a, int axis)
  : AxiswiseFoldOp(a, axis)
{}

OpMeta<Sum> Sum::meta {
  .name = "Sum"
};

template <class T>
inline T foo(T* begin, size_t stride, T* end) {
  T buffer = 0;
  if (stride != 1) {
    for (T* iter = begin; iter != end; iter += stride) {
      buffer += *iter;
    }
  } else {
    buffer = std::accumulate(begin, end, (T) 0, std::plus<T>());
  }
  return buffer;
}

void Sum::run() {
  outputs[0]->malloc();

  switch (inputs[0]->dtype()) {
  case Half: throw std::runtime_error("unsupported dtype: Half");
  case Bool: throw std::runtime_error("unsupported dtype: Bool");

  case Float: runCPU<float>(foo<float>); break;
  case Double: runCPU<double>(foo<double>); break;

  case Sbyte: runCPU<int8_t>(foo<int8_t>); break;
  case Short: runCPU<int16_t>(foo<int16_t>); break;
  case Int: runCPU<int32_t>(foo<int32_t>); break;
  case Long: runCPU<int64_t>(foo<int64_t>); break;

  case Byte: runCPU<uint8_t>(foo<uint8_t>); break;
  case Ushort: runCPU<uint16_t>(foo<uint16_t>); break;
  case Uint: runCPU<uint32_t>(foo<uint32_t>); break;
  case Ulong: runCPU<uint64_t>(foo<uint64_t>); break;

  }
}

}