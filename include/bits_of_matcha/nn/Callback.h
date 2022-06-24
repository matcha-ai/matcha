#pragma once

#include <cstddef>


namespace matcha::nn {

class Net;

struct Callback {

  virtual void init(Net* net) {};

  virtual void onEpochBegin(size_t epoch, size_t max) {};
  virtual void onEpochEnd(size_t epoch, size_t max) {};

  virtual void onBatchBegin(size_t step, size_t max) {};
  virtual void onBatchEnd(size_t step, size_t max) {};

};

}