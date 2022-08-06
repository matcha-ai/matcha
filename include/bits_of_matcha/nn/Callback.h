#pragma once

#include <cstddef>
#include "bits_of_matcha/dataset/Dataset.h"


namespace matcha::nn {

class Net;

struct Callback {

  virtual void onfitInit(Net& net) {};
  virtual void onfitBegin(Net& net, Dataset ds) {};
  virtual void onfitEnd(Net& net) {};

  virtual void onEpochBegin(size_t epoch, size_t max) {};
  virtual void onEpochEnd() {};

  virtual void onBatchBegin(size_t step, size_t max) {};
  virtual void onBatchEnd() {};

  virtual void onPropagateForward(const Instance& instance, const tensor& loss) {};
  virtual void onPropagateBackward(const std::map<tensor*, tensor>& gradients) {};

};

}