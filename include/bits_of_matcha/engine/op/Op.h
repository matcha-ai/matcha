#pragma once

#include "bits_of_matcha/engine/op/Reflection.h"
#include "bits_of_matcha/engine/lambda/Tracer.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/engine/op/typing.h"

#include <initializer_list>
#include <array>


namespace matcha::engine {

class Tensor;

struct Op {
  explicit Op(std::initializer_list<Tensor*> inputs);
  explicit Op(const std::vector<Tensor*>& inputs);
  virtual ~Op();

  std::vector<Tensor*> inputs;
  std::vector<Tensor*> outputs;

  virtual void init();
  virtual void run();

protected:
  Tensor* addOutput(const Frame& frame);
  Tensor* addOutput(const Dtype& dtype, const Shape& shape);
  Tensor* addOutput(Tensor* tensor);

};

void dispatch(Op* op);

template <class Operation, class... Args>
inline std::vector<Tensor*> dispatch(Args... args) {
  auto op = new Operation(args...);
  auto outs = op->outputs;
  dispatch(op);
  return outs;
}

}