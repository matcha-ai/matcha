#pragma once

#include "bits_of_matcha/nn/Loss.h"


namespace matcha::nn {

struct Mse {
  tensor operator()(const tensor& gold, const tensor& pred) const { return mse(gold, pred); }
};

struct Rmse {
  tensor operator()(const tensor& gold, const tensor& pred) const { return rmse(gold, pred); }
};

}