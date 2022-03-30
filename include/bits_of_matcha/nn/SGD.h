#pragma once

#include "bits_of_matcha/nn/Solver.h"

#include <functional>
#include <matcha/tensor>


namespace matcha::nn {

struct LearningRate {
  LearningRate(float constant){};
  LearningRate(std::function<float (unsigned)> scheduling){};
};

struct CategoricalCrossentropy {
  tensor operator()(const tensor& pred, const tensor& gold) {
    return 0;
  }
};

struct SGD {

  LearningRate learning_rate = 1e-3;
  BinaryFn loss;
  size_t epochs = 1;
  size_t batch_size = 32;

  operator Solver();
};


}