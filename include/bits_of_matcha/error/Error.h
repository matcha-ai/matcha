#pragma once

#include <stdexcept>

namespace matcha {

struct Error : std::runtime_error {
  Error(const std::string& message);
  Error(const char* message);
  Error();
};

}