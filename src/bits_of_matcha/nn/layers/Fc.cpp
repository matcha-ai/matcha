#include "bits_of_matcha/nn/layers/Fc.h"
#include "bits_of_matcha/nn/activations.h"

#include <sstream>


namespace matcha::nn {

Layer* Fc::init() {
  static std::map<std::string, UnaryOp> activationFlags = {
    {"relu", relu},
    {"identity", identity},
    {"none", identity},
    {"id", identity},
    {"exp", exp},
//    {"sigmoid", sigmoid},
//    {"sigmoid", softmax},
  };

  static std::set<std::string> bnFlags = {
    "bn",
    "batchnorm",
    "batchnormalization",
    "batchnormalize",
  };

  std::transform(flags.begin(), flags.end(), flags.begin(), tolower);
  std::stringstream ss(flags);
  std::string flag;
  std::set<std::string> flagSet;

  bool bn = false;
  UnaryOp activation = identity;

  while (std::getline(ss, flag, ',')) {
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
    tensor run(const tensor& batch) {
      tensor z = (*linear_)(batch);
      if (bn_);
      return activation_(z);
    }

    std::shared_ptr<Layer> linear_;
    bool bn_;
    UnaryOp activation_;
  };

  auto internal = new Internal;
  internal->linear_ = std::move(linear.internal_);
  internal->bn_ = bn;
  internal->activation_ = activation;
  return internal;
}

tensor Fc::operator()(const tensor& batch) {
  return (*internal_)(batch);
}

}