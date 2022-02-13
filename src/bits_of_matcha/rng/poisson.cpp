#include "bits_of_matcha/rng/poisson.h"

#include <matcha/engine>


namespace matcha {
namespace rng {

Stream poisson() {
  return poisson(1, 0);
}

Stream poisson(float m) {
  return poisson(m, 0);
}

Stream poisson(float m, uint64_t seed) {
  return Stream::fromObject(new engine::rng::Poisson(m, seed));
}

}

namespace engine {
namespace rng {

Poisson::Poisson(float m, uint64_t seed)
  : m_{m}
  , seed_{seed}
  , source_{seed}
  , distribution_{m}
{}

Tensor* Poisson::open(int idx) {
  auto* out = new Tensor(Dtype::Float, {});
  open(idx, out);
  return out;
}

void Poisson::open(int idx, Tensor* tensor) {
  beginOut(tensor);
}

void Poisson::close(Out* out) {
  outs_.erase(std::begin(outs_) + out->id());
}

void Poisson::eval(Out* out) {
  auto* buffer = out->buffer();
  buffer->prepare();
  auto* begin = (float*)buffer->raw();
  auto* end   = begin + out->size();
  for (auto it = begin; it != end; it++) {
    *it = distribution_(source_);
  }
}

bool Poisson::next() {
  for (auto* out: outs_) out->updateStatusChanged();
  return true;
}

bool Poisson::seek(size_t pos) {
  for (auto* out: outs_) out->updateStatusChanged();
  return true;
}

size_t Poisson::tell() const {
  return 0;
}

size_t Poisson::size() const {
  return -1;
}

bool Poisson::eof() const {
  return false;
}


}
}
}
