#pragma once

#include "bits_of_matcha/dataset/Dataset.h"

#include <functional>
#include <variant>


namespace matcha::dataset {

struct Generator {
  Generator(const std::function<Instance ()>& g);
  Generator(const std::function<Instance (size_t pos)>& g);
  operator Dataset();

private:
  std::variant<std::function<Instance ()>, std::function<Instance (size_t)>> g_;

};

}