#include "bits_of_matcha/nn/layers/Affine.h"


namespace matcha::nn {

Layer* Affine::init() {
  struct Internal : Layer {
    unsigned units_;
    bool useBias_;
    tensor kernel_, bias_;

    void init(const tensor& a) override {
      if (a.rank() != 2) throw std::invalid_argument("expected tensor of shape [batchSize, prevLayer]");
      unsigned batch = a.shape()[0];
      unsigned prev = a.shape()[1];
      kernel_ = normal(units_, prev);
      if (useBias_) bias_ = zeros(units_);
    }

    tensor run(const tensor& a) override {
      tensor feed = a;
      feed = kernel_.dot(feed);
      if (useBias_) feed += bias_;
      return feed;
    }
  };
  return new Internal;
}

tensor Affine::operator()(const tensor& a) {
  return (*internal_)(a);
}

}
