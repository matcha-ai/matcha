#pragma once

#include "bits_of_matcha/nn/Callback.h"

#include <iostream>
#include <memory>


namespace matcha::nn {

struct Logger {
  operator std::shared_ptr<Callback>();

  std::ostream& stream = std::cout;
};

}