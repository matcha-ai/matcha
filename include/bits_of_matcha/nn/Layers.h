#pragma once

#include "bits_of_matcha/fn.h"
#include "bits_of_matcha/tensor.h"

#include <string>


namespace matcha::nn {

class Layer {
protected:
  virtual void init(const tensor& feed) = 0;
  virtual tensor run(const tensor& feed) = 0;

public:
  tensor operator()(const tensor& feed) {
    if (!initialized_) init(feed);
    return run(feed);
  }

private:
  bool initialized_;
};


struct Dense {
  unsigned units = 0;
  bool use_bias = true;
  std::string activation = "relu";

  operator UnaryFn();
};

struct Affine {
  unsigned units = 0;
  bool use_bias = true;

  operator UnaryFn();
};

struct Activation {
  std::string activation = "relu";

  operator UnaryFn();
};

}