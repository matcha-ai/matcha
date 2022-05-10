#pragma once

#include "bits_of_matcha/nn/Layer.h"
#include "bits_of_matcha/nn/Optimizer.h"
#include "bits_of_matcha/nn/Callback.h"
#include "bits_of_matcha/nn/callbacks/CoutLogger.h"
#include "bits_of_matcha/dataset/Dataset.h"
#include "bits_of_matcha/Flow.h"


namespace matcha::nn {


class Net {
public:
  void fit(Dataset ds);

public:
  // Sequential API

  Net(std::initializer_list<UnaryOp> sequence);
  Net(const std::vector<UnaryOp>& sequence);

public:
  // Functional API

  Net(const AnyOp& op);

public:
  Optimizer optimizer;
  std::vector<Callback> callbacks = {
    CoutLogger(),
  };

private:
  AnyOp op_;
  Flow trainFlow_, evalFlow_;

};


}

namespace matcha {
using nn::Net;
}