#pragma once

#include "bits_of_matcha/fn.h"
#include "bits_of_matcha/nn/Solver.h"
#include "bits_of_matcha/data/Dataset.h"
#include "bits_of_matcha/Flow.h"

#include <initializer_list>
#include <vector>



namespace matcha::nn {


struct Sequential : Flow {
  Sequential(std::initializer_list<UnaryFn> layers);

  std::vector<UnaryFn> layers;
  Solver solver;

  void fit(const Dataset& dataset);

private:
  tensor run(const tensor& data);
};


}