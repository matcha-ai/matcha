#pragma once

#include "bits_of_matcha/fn.h"
#include "bits_of_matcha/tensor.h"

#include <string>


namespace matcha::nn {

class Layer {
public:
  Layer();
  tensor operator()(const tensor& x);

protected:
  virtual void init(const tensor& x);
  virtual tensor run(const tensor& x) = 0;
  virtual tensor fit(const tensor& x);

private:
  bool initialized_;
};

class Flatten : public Layer {
  tensor run(const tensor& x) override;
};

class Affine : public Layer {
public:
  Affine(unsigned units, bool use_bias = true);

private:
  unsigned units;
  bool use_bias;
  tensor kernel, bias;

  void init(const tensor& x) override;
  tensor run(const tensor& x) override;

};

tensor relu(const tensor& x);

class Relu : public Layer {
  tensor run(const tensor& x);
};

class BatchNormalization : public Layer {
  tensor mAvg, sdAvg;
  float momentum = .9;

  void init(const tensor& x) override;
  tensor run(const tensor& x) override;
  tensor fit(const tensor& x) override;
};

}