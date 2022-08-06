#include "bits_of_matcha/engine/ops/ViewRead.h"
#include "bits_of_matcha/engine/utils/stdVector.h"

#include <queue>


namespace matcha::engine::ops {

ViewRead::ViewRead(engine::Tensor* source,
                   const std::vector<engine::Tensor*>& idxs)
  : Op(cat(std::vector<Tensor*>{source}, idxs))
{
  std::vector<unsigned> dims;
  std::reverse_copy(source->shape().begin(), source->shape().end(), std::back_inserter(dims));

  for (auto&& idx: idxs) {
    if (idx->rank() > 1)
      throw std::invalid_argument(
      "the rank of tensor indices can be at most 1");

    if (dims.empty())
      throw std::out_of_range("invalid index to scalar value");

    if (idx->rank() == 0) {
      dims.pop_back();
    } else {
      dims.back() = idx->size();
    }
  }

  std::reverse(dims.begin(), dims.end());
  addOutput(Float, dims);
}

Reflection<ViewRead> ViewRead::reflection {
  .name = "ViewRead",
};

void ViewRead::run() {
  outputs[0]->malloc();
}

}