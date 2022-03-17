#pragma once

#include "bits_of_matcha/tensor.h"

#include <map>


namespace matcha {

class Instance {
public:
  explicit Instance(std::map<std::string, tensor> data);
  tensor operator[](const std::string& key);
  std::vector<std::string> keys();

private:
  std::map<std::string, tensor> data_;
};

}