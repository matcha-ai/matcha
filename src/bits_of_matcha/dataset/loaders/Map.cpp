#include "bits_of_matcha/dataset/loaders/Map.h"

#include <utility>
#include "bits_of_matcha/engine/dataset/Dataset.h"


namespace matcha::dataset {

Map::Map(Dataset  dataset,
         const std::function<Instance(const Instance&)>& function)
  : dataset_(std::move(dataset))
  , callback_(function)
{}

Map::Map(Dataset  dataset,
         const std::function<void (Instance&)>& function)
  : dataset_(std::move(dataset))
  , callback_(function)
{}

Map::operator Dataset() {
  struct Internal : engine::Dataset {
    matcha::Dataset dataset_;
    Callback callback_;

    Internal(matcha::Dataset  ds, Callback  callback)
      : dataset_(std::move(ds))
      , callback_(std::move(callback))
    {}

    Instance get() override {
      Instance i = dataset_.get();

      if (std::holds_alternative<CbModifying>(callback_)) {
        std::get<CbModifying>(callback_)(i);
        return i;
      }
//      if (std::holds_alternative<CbReturning>(callback_)) {
//        return std::get<CbReturning>(callback_)(i);
//      }
      throw std::runtime_error("invalid map variant");
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

  return ref(new Internal(dataset_, callback_));
}

}
