#pragma once

#include <cstddef>
#include "bits_of_matcha/dataset/Dataset.h"


namespace matcha::nn {

class Net;

struct Callback {

  virtual void onTrainBegin(Net& net, Dataset ds) {};
  virtual void onTrainEnd(Net& net, Dataset ds) {};

  virtual void onEpochBegin(size_t epoch, size_t max) {};
  virtual void onEpochEnd(size_t epoch, size_t max) {};

  virtual void onBatchBegin(size_t step, size_t max) {};
  virtual void onBatchEnd(size_t step, size_t max) {};

};

}