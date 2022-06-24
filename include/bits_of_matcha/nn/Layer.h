#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/ops.h"
#include "bits_of_matcha/random.h"
#include "bits_of_matcha/print.h"
#include "bits_of_matcha/nn/Net.h"

#include <memory>
#include <stack>


namespace matcha::nn {

class Net;

class Layer {
public:
  Layer();
  tensor operator()(const tensor& a);

protected:
  virtual void init(const tensor& a);
  virtual tensor run(const tensor& a) = 0;


  Net::Params params;

private:
  bool initialized_ = false;

  Net* net();

  static thread_local std::stack<Net*> netStack_;
  friend class Net;
};

}