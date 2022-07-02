#pragma once

#include <stdexcept>

namespace matcha {

struct Error : std::runtime_error {
  explicit Error(const std::string& message);
  explicit Error(const char* message);
  explicit Error();
};

}