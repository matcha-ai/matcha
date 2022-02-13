#include "bits_of_matcha/rng/uniform.h"

#include <matcha/engine>


namespace matcha {
namespace rng {

Stream uniform() {
  return uniform(0, 1, 0);
}

Stream uniform(uint64_t seed) {
  return uniform(0, 1, seed);
}

Stream uniform(float lo, float hi) {
  return uniform(lo, hi, 0);
}

Stream uniform(float lo, float hi, uint64_t seed) {
  return Stream::fromObject(new engine::rng::Uniform(lo, hi, seed));
}

}

namespace engine {
namespace rng {

Uniform::Uniform(float lo, float hi, uint64_t seed)
  : lo_{lo}
  , hi_{hi}
  , seed_{seed}
  , source_{seed}
  , distribution_{lo, hi}
{}

bool Uniform::eof() const {
  return false;
}

Tensor* Uniform::open(int idx) {
  auto* out = new Tensor(Dtype::Float, {});
  open(idx, out);
  return out;
}

void Uniform::open(int idx, Tensor* tensor) {
  beginOut(tensor);
}

void Uniform::close(Out* out) {
  outs_.erase(std::begin(outs_) + out->id());
}

void Uniform::eval(Out* out) {
  auto* buffer = out->buffer();
  buffer->prepare();
  auto* begin = (float*)buffer->raw();
  auto* end   = begin + out->size();
  for (auto it = begin; it != end; it++) {
    *it = distribution_(source_);
  }
}

bool Uniform::next() {
  for (auto* out: outs_) out->updateStatusChanged();
  return true;
}

bool Uniform::seek(size_t pos) {
  for (auto* out: outs_) out->updateStatusChanged();
  return true;
}

size_t Uniform::tell() const {
  return 0;
}

size_t Uniform::size() const {
  return -1;
}


}
}
}
