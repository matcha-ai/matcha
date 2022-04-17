#include "bits_of_matcha/engine/flow/graph/TensorMask.h"


namespace matcha::engine {

TensorMask::TensorMask(Graph& graph, bool defaultValue)
  : TensorDict(graph, defaultValue)
{}

TensorMask::TensorMask(Graph* graph, bool defaultValue)
  : TensorDict(graph, defaultValue)
{}

TensorMask TensorMask::operator~() const {
  TensorMask result(graph_);
  std::transform(
    begin(), end(),
    result.begin(),
    std::logical_not()
  );
  return result;
}

TensorMask TensorMask::operator&(const TensorMask& mask) const {
  TensorMask result(graph_);
  std::transform(
    begin(), end(),
    mask.begin(),
    result.begin(),
    std::logical_and()
  );
  return result;
}

TensorMask TensorMask::operator|(const TensorMask& mask) const {
  TensorMask result(graph_);
  std::transform(
    begin(), end(),
    mask.begin(),
    result.begin(),
    std::logical_or()
  );
  return result;
}

TensorMask& TensorMask::operator&=(const TensorMask& mask) {
  *this = *this & mask;
  return *this;
}

TensorMask& TensorMask::operator|=(const TensorMask& mask) {
  *this = *this | mask;
  return *this;
}

size_t TensorMask::count() const {
  return std::count(begin(), end(), true);
}

std::vector<Tensor*> TensorMask::get() const {
  std::vector<Tensor*> result;
  for (int i = 0; i < size(); i++) {
    if (values_[i]) result.push_back(graph_->tensors[i]);
  }
  return result;
}

std::vector<Tensor*> TensorMask::rget() const {
  std::vector<Tensor*> result;
  for (int i = (int) size() - 1; i >= 0; i--) {
    if (values_[i]) result.push_back(graph_->tensors[i]);
  }
  return result;
}

}
