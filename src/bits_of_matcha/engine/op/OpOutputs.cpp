#include "bits_of_matcha/engine/op/OpOutputs.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/print.h"


namespace matcha::engine {

void OpOutputs::add(Op* op, Tensor* tensor) {
  data_.push_back(tensor);
  if (tensor) tensor->setOp(op);
}

void OpOutputs::add(Op* op, const Frame& frame) {
  auto tensor = new Tensor(frame, op);
  data_.push_back(tensor);
}

void OpOutputs::add(Op* op, const Dtype& dtype, const Shape& shape) {
  add(op, Frame(dtype, shape));
}

Tensor*& OpOutputs::operator[](int idx) {
  return data_[idx];
}

size_t OpOutputs::size() const {
  return data_.size();
}

Tensor** OpOutputs::begin() {
  return &data_[0];
}

Tensor** OpOutputs::end() {
  return begin() + size();
}

bool OpOutputs::any() const {
  return !data_.empty();
}

bool OpOutputs::none() const {
  return data_.empty();
}

OpOutputs::~OpOutputs() {
  for (auto out: data_) {
    if (!out) continue;
    out->setOp(nullptr);
  }
}

std::vector<Tensor*>& OpOutputs::stdVector() {
  return data_;
}

const std::vector<Tensor*>& OpOutputs::stdVector() const {
  return data_;
}

}