#pragma once

#include "bits_of_matcha/fn.h"

namespace matcha {


class Sequence {
public:
  Sequence(std::initializer_list<UnaryFn> phases);
  tensor operator()(const tensor& a);

private:
  std::vector<UnaryFn> phases_;
};


}