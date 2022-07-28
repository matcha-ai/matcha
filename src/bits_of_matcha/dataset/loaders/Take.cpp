#include "bits_of_matcha/dataset/loaders/Take.h"
#include "bits_of_matcha/engine/dataset/Dataset.h"


namespace matcha::dataset {

Take::Take(const Dataset& dataset, size_t limit)
  : dataset_(dataset)
  , limit_(limit)
{}

Take::operator Dataset() {
  struct Internal : engine::Dataset {
    matcha::Dataset dataset_;
    size_t limit_;
    size_t offset_;

    Internal(const matcha::Dataset& ds, size_t limit)
      : dataset_(ds)
      , limit_(limit)
    {
      offset_ = dataset_.tell();
    }

    Instance get() override {
      return dataset_.get();
    }

    size_t size() const override {
      return std::min(limit_, dataset_.size() - offset_);
    }

    size_t tell() const override {
      return dataset_.tell() - offset_;
    }

    void seek(size_t pos) override {
      dataset_.seek(offset_ + pos);
    }
  };

  return ref(new Internal(dataset_, limit_));
}

}