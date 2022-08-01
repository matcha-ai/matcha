#pragma once

#include "bits_of_matcha/ops.h"

#include <cstddef>


namespace matcha::engine {
class Tensor;
class Flow;
class Dataset;
}

namespace matcha {
class tensor;
class Flow;
class Dataset;
class InstanceIterator;
class Instance;
class fn;
}

namespace matcha {

class Engine final {
public:
  static tensor ref(engine::Tensor* internal);
  static engine::Tensor* deref(const tensor& external);
  static engine::Tensor* deref(const tensor* external);
  static engine::Tensor* unref(tensor& external);
  static engine::Tensor* unref(tensor* external);

  static Dataset ref(engine::Dataset* internal);
  static engine::Dataset* deref(const Dataset& external);
  static engine::Dataset* deref(const Dataset* external);
  static engine::Dataset* unref(Dataset& external);
  static engine::Dataset* unref(Dataset* external);

  static InstanceIterator makeInstanceIterator(engine::Dataset* ds, size_t pos);

};

}