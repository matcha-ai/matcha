#pragma once

#include "bits_of_matcha/engine/op/OpInputs.h"
#include "bits_of_matcha/engine/op/OpOutputs.h"
#include "bits_of_matcha/engine/op/OpCtx.h"
#include "bits_of_matcha/engine/op/OpMeta.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/engine/autograd/BackCtx.h"

#include <initializer_list>
#include <array>


namespace matcha::engine {

class Tensor;

struct Op {
  Op(std::initializer_list<Tensor*> inputs);
  Op(const std::vector<Tensor*>& inputs);

  OpInputs inputs;
  OpOutputs outputs;
  OpCtx& ctx();

  virtual void init();
  virtual void run();

protected:
  OpCtx ctx_;
};

void collect(Op* op);

}