#pragma once

#include "bits_of_matcha/engine/op/OpInputs.h"
#include "bits_of_matcha/engine/op/OpOutputs.h"
#include "bits_of_matcha/engine/op/OpMeta.h"
#include "bits_of_matcha/engine/flow/Tracer.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/engine/op/typing.h"

#include <initializer_list>
#include <array>


namespace matcha::engine {

class Tensor;

struct Op {
  Op(std::initializer_list<Tensor*> inputs);
  Op(const std::vector<Tensor*>& inputs);
  virtual ~Op();

  OpInputs inputs;
  OpOutputs outputs;

  virtual void init();
  virtual void run();

};

void send(Op* op);

}