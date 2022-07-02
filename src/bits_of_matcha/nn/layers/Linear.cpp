#include "bits_of_matcha/nn/layers/Linear.h"

#include <cmath>


namespace matcha::nn {

Layer* Linear::init() {
  struct Internal : Layer {
    unsigned units_;
    bool useBias_;
    tensor kernel_, bias_;
    Generator initializer_;

    void init(const tensor& a) override {
      if (a.rank() != 2) throw std::invalid_argument("expected tensor of shape [batchSize, prevLayer]");
      unsigned batch = a.shape()[0];
      unsigned prev = a.shape()[1];

      kernel_ = initializer_(units_, prev);
      params.add(&kernel_);
      if (useBias_) {
        bias_ = zeros(units_);
        params.add(&bias_);
      }
    }

    tensor run(const tensor& a) override {
      unsigned bs = a.shape()[0];
      tensor z = kernel_.dot(a.reshape(bs, -1, 1)).reshape(bs, -1);
      if (useBias_) z += bias_;
      return z;
    }
  };
  auto internal = new Internal;
  internal->units_ = units;
  internal->useBias_ = useBias;
  internal->initializer_ = initializer;
  return internal;
}

tensor Linear::operator()(const tensor& a) {
  return (*internal_)(a);
}

}
