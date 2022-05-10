#pragma once

#include "bits_of_matcha/dataset/Dataset.h"

namespace matcha::dataset {

struct Take {
  Take(const Dataset& dataset, size_t limit);
  operator Dataset();

private:
  Dataset dataset_;
  size_t limit_;
};

}