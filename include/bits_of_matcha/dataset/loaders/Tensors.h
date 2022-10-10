#pragma once

#include "bits_of_matcha/dataset/Dataset.h"

namespace matcha::dataset {

struct Tensors {
  Tensors(std::vector<tensor>  tensors);
  Tensors(std::initializer_list<tensor> tensors);

  operator Dataset();

private:
  std::vector<tensor> data_;
};

}