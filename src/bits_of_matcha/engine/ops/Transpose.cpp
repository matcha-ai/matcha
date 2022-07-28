#include "bits_of_matcha/engine/ops/Transpose.h"
#include "bits_of_matcha/engine/cpu/kernels/transpose.h"

#include <cblas.h>


namespace matcha::engine::ops {

Transpose::Transpose(Tensor* a)
  : Op{a}
  , iter_(a->shape())
{
  auto& shapeA = a->shape();
  std::vector<unsigned> dims;

  if (shapeA.rank() >= 2) {
    dims = std::vector(shapeA.begin(), shapeA.end());
    std::swap(dims[dims.size() - 1], dims[dims.size() - 2]);
  } else {
    throw std::invalid_argument("can't transpose scalar or vector");
  }

  outputs.add(this, a->dtype(), dims);
}

OpMeta<Transpose> Transpose::meta {
  .name = "Transpose",
};

void Transpose::run() {
  if (iter_.rows == 1 || iter_.cols == 1) {
    outputs[0]->share(inputs[0]);
    return;
  }

  switch (inputs[0]->dtype()) {
  case Float: cpu::transpose<float>(inputs[0]->buffer(), outputs[0]->malloc(), iter_); break;
  case Double: cpu::transpose<double>(inputs[0]->buffer(), outputs[0]->malloc(), iter_); break;
  case Sbyte: cpu::transpose<int8_t>(inputs[0]->buffer(), outputs[0]->malloc(), iter_); break;
  case Short: cpu::transpose<int16_t>(inputs[0]->buffer(), outputs[0]->malloc(), iter_); break;
  case Int: cpu::transpose<int32_t>(inputs[0]->buffer(), outputs[0]->malloc(), iter_); break;
  case Long: cpu::transpose<int64_t>(inputs[0]->buffer(), outputs[0]->malloc(), iter_); break;
  case Byte: cpu::transpose<uint8_t>(inputs[0]->buffer(), outputs[0]->malloc(), iter_); break;
  case Ushort: cpu::transpose<uint16_t>(inputs[0]->buffer(), outputs[0]->malloc(), iter_); break;
  case Uint: cpu::transpose<uint32_t>(inputs[0]->buffer(), outputs[0]->malloc(), iter_); break;
  case Ulong: cpu::transpose<uint64_t>(inputs[0]->buffer(), outputs[0]->malloc(), iter_); break;
  case Bool: cpu::transpose<bool>(inputs[0]->buffer(), outputs[0]->malloc(), iter_); break;
  }
}

}
