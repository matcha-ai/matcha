#pragma once

#include "bits_of_matcha/dataset/Dataset.h"

#include <set>
#include <string>

namespace matcha::dataset {

struct Csv {
  std::string file;
  std::set<std::string> y = {"class", "label", "y", "target"};

  operator Dataset();
};

}