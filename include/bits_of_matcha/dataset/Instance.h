#pragma once

#include "bits_of_matcha/tensor.h"

#include <map>
#include <string>

namespace matcha {

class Instance {
public:

private:
  std::map<std::string, tensor> data_;
};

}