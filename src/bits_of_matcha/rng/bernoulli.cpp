#include "bits_of_matcha/rng/bernoulli.h"
#include "bits_of_matcha/engine/tensor.h"
#include "bits_of_matcha/engine/input.h"
#include "bits_of_matcha/engine/stream.h"
#include "bits_of_matcha/engine/stream.h"
#include "bits_of_matcha/engine/flowloader.h"
#include "bits_of_matcha/engine/flowsaver.h"
#include "bits_of_matcha/stream.h"

#include <matcha/device>


namespace matcha {
namespace rng {

Stream bernoulli() {
  return bernoulli(0.5, 0);
}

Stream bernoulli(float p) {
  return bernoulli(p, 0);
}

Stream bernoulli(float p, uint64_t seed) {
  return Stream::fromObject(new engine::rng::Bernoulli(p, seed));
}

}

namespace engine {
namespace rng {

Bernoulli::Bernoulli(float p, uint64_t seed)
  : p_{p}
  , seed_{seed}
  , source_{seed}
  , distribution_{p}
{}

Tensor* Bernoulli::open(int idx) {
  auto* out = new Tensor(Dtype::Float, {});
  open(idx, out);
  return out;
}

void Bernoulli::open(int idx, Tensor* tensor) {
  beginOut(tensor);
}

void Bernoulli::close(Out* out) {
  outs_.erase(std::begin(outs_) + out->id());
}

void Bernoulli::eval(Out* out) {
  auto* buffer = out->buffer();
  buffer->prepare();
  auto* begin = (float*)buffer->raw();
  auto* end   = begin + out->size();
  for (auto it = begin; it != end; it++) {
    *it = distribution_(source_);
  }
}

bool Bernoulli::next() {
  for (auto* out: outs_) out->updateStatusChanged();
  return true;
}

bool Bernoulli::seek(size_t pos) {
  for (auto* out: outs_) out->updateStatusChanged();
}

size_t Bernoulli::tell() const {
  return 0;
}

size_t Bernoulli::size() const {
  return -1;
}

bool Bernoulli::eof() const {
  return false;
}


}
}
}
