#include "bits_of_matcha/nn/layers/Fc.h"
#include "bits_of_matcha/nn/layers/BatchNorm.h"
#include "bits_of_matcha/nn/activations.h"

#include <sstream>


namespace matcha::nn {

Layer* Fc::init() {
  static std::map<std::string, UnaryOp> activation_flags = {
    {"relu", nn::relu},
    {"identity", identity},
    {"none", identity},
    {"id", identity},
    {"exp", exp},
    {"sigmoid", nn::sigmoid},
    {"softmax", nn::softmax},
    {"tanh", nn::tanh},
  };

  static std::set<std::string> bn_flags = {
    "bn",

    "bnorm",
    "bnormalize",
    "bnormalization",

    "batchnorm",
    "batchnormalize",
    "batchnormalization",
  };

  static std::set<std::string> nobias_flags = {
    "nobias",
  };

  std::transform(flags.begin(), flags.end(), flags.begin(), tolower);
  std::stringstream ss(flags);
  std::string flag;
  std::set<std::string> flagSet;

  bool use_bias = true;
  bool bn = false;
  UnaryOp activation = identity;

  while (std::getline(ss, flag, ',')) {
    // trim leading and trailing spaces
    flag.erase(std::remove_if(flag.begin(), flag.end(), isspace), flag.end());

    if (bn_flags.contains(flag)) {
      bn = true;
      continue;
    }

    if (nobias_flags.contains(flag)) {
      use_bias = false;
      continue;
    }

    if (activation_flags.contains(flag)) {
      activation = activation_flags[flag];
      continue;
    }

    throw std::invalid_argument("couldn't parse Fc configuration");
  }

  use_bias &= !bn;

  auto linear = Linear {
    .units = units,
    .use_bias = use_bias,
  };


  struct Internal : Layer {
    tensor run(const tensor& batch) override {
      tensor z = (*linear_)(batch);
      if (bn_) z = (*bn_)(z);
      return activation_(z);
    }

    std::shared_ptr<Layer> linear_;
    std::shared_ptr<Layer> bn_;
    UnaryOp activation_;
  };

  auto internal = new Internal;
  internal->linear_ = std::move(linear.internal_);
  internal->bn_ = bn ? BatchNorm{}.internal_ : nullptr;
  internal->activation_ = activation;
  return internal;
}

tensor Fc::operator()(const tensor& batch) {
  return (*internal_)(batch);
}

}