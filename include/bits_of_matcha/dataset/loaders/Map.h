#pragma once

#include "bits_of_matcha/dataset/Dataset.h"

#include <functional>


namespace matcha::dataset {

struct Map {
  Map(const Dataset& dataset, const std::function<Instance (const Instance&)>& function);
  operator Dataset();

private:
  Dataset dataset_;
  std::function<Instance (const Instance&)> function_;
};

}
