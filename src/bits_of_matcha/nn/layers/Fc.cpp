#include "bits_of_matcha/nn/layers/Fc.h"
#include "bits_of_matcha/nn/layers/BatchNorm.h"
#include "bits_of_matcha/nn/activations.h"

#include <sstream>


namespace matcha::nn {

Layer* Fc::init() {
  static std::map<std::string, UnaryOp> activationFlags = {
    {"relu", nn::relu},
    {"identity", identity},
    {"none", identity},
    {"id", identity},
    {"exp", exp},
    {"sigmoid", nn::sigmoid},
    {"softmax", nn::softmax},
    {"tanh", nn::tanh},
  };

  static std::set<std::string> bnFlags = {
    "bn",

    "bnorm",
    "bnormalize",
    "bnormalization",

    "batchnorm",
    "batchnormalize",
    "batchnormalization",
  };

  std::transform(flags.begin(), flags.end(), flags.begin(), tolower);
  std::stringstream ss(flags);
  std::string flag;
  std::set<std::string> flagSet;

  bool bn = false;
  UnaryOp activation = identity;

  while (std::getline(ss, flag, ',')) {
    // trim leading and trailing spaces
    flag.erase(std::remove_if(flag.begin(), flag.end(), isspace), flag.end());

    if (bnFlags.contains(flag)) {
      bn = true;
      continue;
    }

    if (activationFlags.contains(flag)) {
      activation = activationFlags[flag];
      continue;
    }

    throw std::invalid_argument("couldn't parse Fc configuration");
  }

  auto linear = Linear {
    .units = units,
    .useBias = !bn
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