#include "bits_of_matcha/rng/normal.h"

#include <matcha/engine>


namespace matcha {
namespace rng {

Stream normal() {
  return normal(0, 1, 0);
}

Stream normal(uint64_t seed) {
  return normal(0, 1, seed);
}

Stream normal(float m, float sd) {
  return normal(m, sd, 0);
}

Stream normal(float m, float sd, uint64_t seed) {
  return Stream::fromObject(new engine::rng::Normal(m, sd, seed));
}

}

namespace engine {
namespace rng {

Normal::Normal(float m, float sd, uint64_t seed)
  : m_{m}
  , sd_{sd}
  , seed_{seed}
  , source_{seed}
  , distribution_{m, sd}
{}

Tensor* Normal::open(int idx) {
  auto* out = new Tensor(Dtype::Float, {});
  open(idx, out);
  return out;
}

void Normal::open(int idx, Tensor* tensor) {
  beginOut(tensor);
}

void Normal::close(Out* out) {
  outs_.erase(std::begin(outs_) + out->id());
}

void Normal::eval(Out* out) {
  auto* buffer = out->buffer();
  buffer->prepare();
  auto* begin = (float*)buffer->raw();
  auto* end   = begin + out->size();
  for (auto it = begin; it != end; it++) {
    *it = distribution_(source_);
  }
}

bool Normal::next() {
  for (auto* out: outs_) out->updateStatusChanged();
  return true;
}

bool Normal::seek(size_t pos) {
  for (auto* out: outs_) out->updateStatusChanged();
  return true;
}

size_t Normal::tell() const {
  return 0;
}

size_t Normal::size() const {
  return -1;
}

bool Normal::eof() const {
  return false;
}

}
}
}
