#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/fn.h"

#include <functional>
#include <map>


namespace matcha {

class Grad {
public:
  std::map<tensor, tensor> propagate(const tensor& delta);


};

}