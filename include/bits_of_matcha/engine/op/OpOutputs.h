#pragma once

#include "bits_of_matcha/Frame.h"
#include "bits_of_matcha/engine/op/OpInputs.h"

#include <vector>

namespace matcha::engine {

class Tensor;
class Op;

class OpOutputs {
public:
  void add(Op* op, Tensor* tensor);
  void add(Op* op, const Frame& frame);
  void add(Op* op, const Dtype& dtype, const Shape& shape);

  Tensor*& operator[](int idx);
  size_t size() const;

  Tensor** begin();
  Tensor** end();

private:
  std::vector<Tensor*> data_;

};


}
