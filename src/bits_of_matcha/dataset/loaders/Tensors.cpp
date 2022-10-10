#include "bits_of_matcha/dataset/loaders/Tensors.h"

#include <utility>
#include "bits_of_matcha/engine/dataset/Dataset.h"


namespace matcha::dataset {

Tensors::Tensors(std::vector<tensor>  tensors)
  : data_(std::move(tensors))
{}

Tensors::Tensors(std::initializer_list<tensor> tensors)
  : data_(tensors)
{}

Tensors::operator Dataset() {
  struct Internal : engine::Dataset {
    std::vector<tensor> data_;
    size_t pos_ = 0;

    Internal(std::vector<tensor>  data) : data_(std::move(data)) {}

    Instance get() override {
      return {{{"0", data_[pos_++]}}};
    }

    size_t size() const override {
      return data_.size();
    }

    size_t tell() const override {
      return pos_;
    }

    void seek(size_t pos) override {
      pos_ = pos;
    }
  };

  return ref(new Internal(data_));
}

}