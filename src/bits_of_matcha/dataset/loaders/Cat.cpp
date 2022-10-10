#include "bits_of_matcha/dataset/loaders/Cat.h"

#include <utility>
#include "bits_of_matcha/engine/dataset/Dataset.h"


namespace matcha::dataset {

Cat::Cat(std::vector<Dataset>  datasets)
  : datasets_(std::move(datasets))
{}

Cat::operator Dataset() {
  struct Internal : engine::Dataset {
    std::vector<matcha::Dataset> datasets_;
    size_t iter1_, iter2_, size_, pos_;

    explicit Internal(std::vector<matcha::Dataset>  datasets)
      : datasets_(std::move(datasets))
      , iter1_(0)
      , iter2_(0)
      , size_(0)
    {
      for (auto&& ds: datasets_) size_ += ds.size();
    }

    Instance get() override {
      auto& ds = datasets_[iter1_];
      ds.seek(iter2_);
      auto i = ds.get();
      pos_++;
      iter2_++;
      if (iter2_ >= ds.size()) {
        iter1_++;
        iter2_ = 0;
      }
      return i;
    }

    size_t size() const override {
      return size_;
    }

    size_t tell() const override {
      return pos_;
    }

    void seek(size_t pos) override {
      if (pos == pos_) return;
      pos_ = 0;
      iter1_ = 0;
      while (true) {
        pos_ += datasets_[iter1_].size();
        if (pos_ > pos) break;
        iter1_++;
      }
      iter2_ = pos - (pos_ - datasets_[iter1_].size());
      pos_ = pos;
    }
  };

  return ref(new Internal(datasets_));
}

}
