#pragma once

#include "bits_of_matcha/dataset/Dataset.h"

#include <set>
#include <string>

namespace matcha::dataset {

struct Csv {
  std::string file;
  std::set<std::string> classification_tags = {
    "class", "label"
  };
  std::set<std::string> regression_tags = {
    "target", "y"
  };

  operator Dataset();
};

}