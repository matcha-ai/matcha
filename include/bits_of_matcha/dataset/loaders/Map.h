#pragma once

#include "bits_of_matcha/dataset/Dataset.h"

#include <functional>


namespace matcha::dataset {

struct Map {
  Map(Dataset  dataset,
      const std::function<Instance (const Instance&)>& function);
  Map(Dataset  dataset,
      const std::function<void (Instance&)>& function);

  operator Dataset();

private:
  using CbModifying = std::function<void (Instance&)>;
  using CbReturning = std::function<Instance (const Instance&)>;
  using Callback = std::variant<CbModifying, CbReturning>;

  Callback callback_;
  Dataset dataset_;
};

}
