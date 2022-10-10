#include "bits_of_matcha/dataset/loaders/Generator.h"
#include "bits_of_matcha/engine/dataset/Dataset.h"

#include <limits>
#include <utility>


namespace matcha::dataset {

Generator::Generator(const std::function<Instance()>& g)
  : g_(g)
{}

Generator::Generator(const std::function<Instance(size_t)>& g)
  : g_(g)
{}

Generator::operator Dataset() {
  struct Indexed : engine::Dataset {
    std::function<Instance (size_t)> g_;
    size_t pos_ = 0;

    Indexed(std::function<Instance (size_t)>  g) : g_(std::move(g)) {}

    Instance get() override {
      return g_(pos_++);
    }

    void seek(size_t pos) {
      pos_ = pos;
    }

    size_t size() const override {
      return std::numeric_limits<size_t>::max();
    }

    size_t tell() const override {
      return pos_;
    }
  };

  std::function<Instance (size_t)> callback;
  if (std::holds_alternative<std::function<Instance ()>>(g_)) {
    auto unindexed = std::get<std::function<Instance ()>>(g_);
    callback = [=](size_t) { return unindexed(); };
  } else {
    callback = std::get<std::function<Instance (size_t)>>(g_);
  }

  return ref(new Indexed(callback));
}

}