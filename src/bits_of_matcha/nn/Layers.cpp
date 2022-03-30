#include "bits_of_matcha/nn/Layers.h"
#include "bits_of_matcha/nn/Net.h"

#include <matcha/tensor>


namespace matcha::nn {

Layer::Layer()
  : initialized_{false}
{}

tensor Layer::operator()(const tensor& x) {
  bool training = Net::ctx()->training;
  if (!initialized_) init(x);
  return training ? fit(x) : run(x);
}

void Layer::init(const tensor& x) {}
tensor Layer::fit(const tensor& x) {
  return run(x);
}

tensor Flatten::run(const tensor& x) {
  return x.reshape(x.shape()[0], -1);
}

Affine::Affine(unsigned int units, bool use_bias)
  : units{units}
  , use_bias{use_bias}
{}

void Affine::init(const tensor& x) {
  unsigned batch = x.shape()[0];
  unsigned prev = x.shape()[-1];

  rng::Uniform init{};
  kernel = init(units, prev);
//  Net::Ctx::train(kernel);

  if (use_bias) {
    bias = tensor::zeros(units);
//    Net::Ctx::train(bias);
  }

}

tensor Affine::run(const tensor& x) {
  return use_bias ? x.map(kernel, bias) : x.map(kernel);
}

tensor relu(const tensor& x) {
  return fn::max(x, 0);
}

tensor Relu::run(const tensor& x) {
  return relu(x);
}

void BatchNormalization::init(const tensor& x) {
  std::vector<unsigned> dims(x.shape().begin() + 1, x.shape().end());
  mAvg = tensor::zeros(Shape(dims));
  sdAvg = tensor::ones(Shape(dims));
}

tensor BatchNormalization::run(const tensor& x) {
  print("run");
  return (x - mAvg) / sdAvg;
}

tensor BatchNormalization::fit(const tensor& x) {
  print("fit");
  tensor m = fn::sum(x) / x.shape()[0];
  tensor sd = fn::sum((x - m).pow(2)).nrt(2) / x.shape()[0];
  mAvg = momentum * mAvg + (1 - momentum) * m;
  sdAvg = momentum * sdAvg + (1 - momentum) * sd;
  return (x - m) / sd;
}

}