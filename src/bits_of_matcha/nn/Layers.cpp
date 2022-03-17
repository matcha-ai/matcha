#include "bits_of_matcha/nn/Layers.h"

#include <matcha/tensor>


namespace matcha::nn {


Affine::operator UnaryFn() {
  struct Internal : Layer {
    tensor kernel, bias;

    Internal(unsigned nodes, bool use_bias)
      : nodes{nodes}
      , use_bias{use_bias}
    {}

    unsigned nodes;
    bool use_bias;

    void init(const tensor& feed) override {
    }

    tensor run(const tensor& feed) override {
      tensor y = feed.map(kernel);
      if (use_bias) y += bias;
      return y;
    }
  };

  return Internal(units, use_bias);
}

Dense::operator UnaryFn() {
  struct Internal : Layer {
    tensor kernel, bias;

    Internal(unsigned nodes, bool use_bias, const std::string& activation)
      : nodes{nodes}
      , use_bias{use_bias}
      , activation{}
    {}

    unsigned nodes;
    bool use_bias;
    UnaryFn activation;

    void init(const tensor& feed) override {
    }

    tensor run(const tensor& feed) override {
      tensor y = feed.map(kernel);
      if (use_bias) y += bias;
      return activation(y);
    }
  };

  return Internal(units, use_bias, activation);
}

Activation::operator UnaryFn() {
  if (activation == "relu") return [](const tensor& x) { return fn::max(x, 0); };
  if (activation == "tanh") return [](const tensor& x) { return fn::max(x, 0); };
  throw std::invalid_argument("unknown activation");
}


}