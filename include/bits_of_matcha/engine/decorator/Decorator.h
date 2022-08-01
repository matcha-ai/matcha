#pragma once

#include "bits_of_matcha/ops.h"
#include "bits_of_matcha/fn.h"

namespace matcha::engine {

class Transform {
public:
  explicit Transform(const fn& preimage);
  explicit Transform();

private:
  fn preimage_;
};

}