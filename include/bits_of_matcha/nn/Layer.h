#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/ops.h"
#include "bits_of_matcha/random.h"
#include "bits_of_matcha/print.h"

#include <memory>


namespace matcha::nn {

class Layer {
public:
  Layer();
  tensor operator()(const tensor& a);

protected:
  virtual void init(const tensor& a);
  virtual tensor run(const tensor& a) = 0;

private:
  bool initialized_ = false;

};

}