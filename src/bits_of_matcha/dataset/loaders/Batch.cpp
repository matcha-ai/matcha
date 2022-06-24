#include "bits_of_matcha/dataset/loaders/Batch.h"
#include "bits_of_matcha/engine/dataset/Dataset.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/engine/tensor/factories.h"
#include "bits_of_matcha/print.h"
#include "bits_of_matcha/ops.h"

#include <vector>
#include <map>


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
      if (limit_ == 0)
        throw std::invalid_argument("batch size limit must not be 0");
//      offset_ = dataset_.tell();
      if (limit_ > std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("batch size limit is out of range");
    }

    Instance get() override {
      std::map<std::string, std::vector<tensor>> source;

      size_t size;
      for (size = 0; size < limit_; size++) {
        Instance i = dataset_.get();
        if (!i) {
          size--;
          break;
        }
        for (auto&& [key, t]: i) {
          source[key].push_back(t);
        }
      }

      if (!size) return Instance{};

      const char* err = "batching dataset failed";

      Instance target;
      for (auto&& [key, ts]: source) {
        if (ts.size() != size) throw std::runtime_error(err);
        target[key] = stack(ts);
      }
      return target;
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
