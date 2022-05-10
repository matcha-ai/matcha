#include "bits_of_matcha/dataset/loaders/Map.h"
#include "bits_of_matcha/engine/dataset/Dataset.h"


namespace matcha::dataset {

Map::Map(const Dataset& dataset, const std::function<Instance(const Instance&)>& function)
  : dataset_(dataset)
  , function_(function)
{}

Map::operator Dataset() {
  struct Internal : engine::Dataset {
    matcha::Dataset dataset_;
    std::function<Instance (const Instance&)> function_;

    Internal(const matcha::Dataset& ds, const std::function<Instance (const Instance&)>& function)
      : dataset_(ds)
      , function_(function)
    {}

    Instance get() override {
      return function_(dataset_.get());
    }

    size_t size() const override {
      return dataset_.size();
    }

    size_t tell() const override {
      return dataset_.tell();
    }

    void seek(size_t pos) override {
      dataset_.seek(pos);
    }
  };

  return ref(new Internal(dataset_, function_));
}

}
