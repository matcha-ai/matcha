#include "bits_of_matcha/engine/op/OpInputs.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"


namespace matcha::engine {

OpInputs::OpInputs(const std::vector<Tensor*>& inputs)
  : data_{inputs}
{
  for (auto i: data_) i->req();
}

size_t OpInputs::size() const {
  return data_.size();
}

Tensor** OpInputs::begin() {
  return &data_[0];
}

Tensor** OpInputs::end() {
  return begin() + data_.size();
}

Tensor*& OpInputs::operator[](int idx) {
  return data_[idx];
}

bool OpInputs::any() const {
  return !data_.empty();
}

bool OpInputs::none() const {
  return data_.empty();
}

}