#include "bits_of_matcha/nn/layers/BatchNorm.h"


namespace matcha::nn {

tensor BatchNorm::operator()(const tensor& batch) {
  return (*internal_)(batch);
}

Layer* BatchNorm::init() {
  struct Internal : Layer {
    tensor gamma, beta;

    void init(const tensor& batch) override {
      gamma = 1.;
      params.add(&gamma);

      beta = 0.;
      params.add(&beta);
    }

    tensor run(const tensor& batch) override {
      auto bsize = (float) batch.shape()[0];
      if (bsize == 1) return batch;

      tensor means = mean(batch, 0);
      tensor sdevs = stdev(batch, 0);
      tensor norms = (batch - means) / sdevs;
      return gamma * norms + beta;
    }
  };

  return new Internal;
}

}