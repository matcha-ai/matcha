#include "bits_of_matcha/nn/layers/Fc.h"


namespace matcha::nn {

UnaryOp Fc::init() {
  struct Internal : Layer {
    Affine affine_;
    Activation activation_;

    Internal(const Affine& affine, const Activation& activation)
      : affine_(affine)
      , activation_(activation)
    {}

    tensor run(const tensor& a) override {
      return a;
    }
  };

  Affine affine {
    .units = units,
    .useBias = useBias
  };
  return Internal(affine, activation);
}

tensor Fc::operator()(const tensor& a) {
  return op_(a);
}

}