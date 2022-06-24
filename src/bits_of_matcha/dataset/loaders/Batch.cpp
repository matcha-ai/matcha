#include "bits_of_matcha/dataset/loaders/Batch.h"
#include "bits_of_matcha/engine/dataset/Dataset.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/engine/tensor/factories.h"
#include "bits_of_matcha/print.h"


namespace matcha::dataset {

Batch::Batch(const Dataset& dataset, size_t limit)
  : dataset_(dataset)
  , limit_(limit)
{}

Batch::operator Dataset() {
  struct Internal : engine::Dataset {
    matcha::Dataset dataset_;
    size_t limit_;
    size_t offset_;

    Internal(const matcha::Dataset& ds, size_t limit)
      : dataset_(ds)
      , limit_(limit)
    {
//      offset_ = dataset_.tell();
    }

    Instance get() override {
      Instance first = dataset_.get();
      Instance result;
      for (auto& a: first);
      for (auto& [key, t]: first) {
        std::vector dims(t.shape().begin(), t.shape().end());
        dims.insert(dims.begin(), limit_);
        Dtype dtype = t.dtype();
        result[key] = zeros(Shape(dims));
//        print(result[key]);
//        result[key] = engine::ref(engine::zeros(dims));
      }
      return result;
    }

    size_t size() const override {
      return dataset_.size() / limit_;
    }

    size_t tell() const override {
      return dataset_.tell() / limit_;
    }

    void seek(size_t pos) override {
      dataset_.seek(pos * limit_);
    }
  };

  return ref(new Internal(dataset_, limit_));
}

}
