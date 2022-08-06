#pragma once

#include "bits_of_matcha/nn/Loss.h"


namespace matcha::nn {

struct Mse {
  tensor operator()(const tensor& expected, const tensor& predicted) const {
    return mse(expected, predicted);
  }
};

struct Rmse {
  tensor operator()(const tensor& expected, const tensor& predicted) const {
    return rmse(expected, predicted);
  }
};

}