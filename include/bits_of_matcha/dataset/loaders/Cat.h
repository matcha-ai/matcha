#pragma once

#include "bits_of_matcha/dataset/Dataset.h"

#include <functional>


namespace matcha::dataset {

struct Cat {
  explicit Cat(std::vector<Dataset>  datasets);
  operator Dataset();

private:
  std::vector<Dataset> datasets_;
};

}
