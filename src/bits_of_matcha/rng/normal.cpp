#include "bits_of_matcha/rng/normal.h"
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

void Normal::reset() {

}

void Normal::shuffle() {

}

bool Normal::eof() const {
  return false;
}

size_t Normal::size() const {
  return std::numeric_limits<size_t>::max();
}

Tensor* Normal::open() {
  auto* out = new Tensor(Dtype::Float, {});
  open(out);
  return out;
}

void Normal::open(Tensor* tensor) {
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

/*

const NodeLoader* Normal::getLoader() const {
  return loader();
}

const NodeLoader* Normal::loader() {
  static NodeLoader nl = {
    .type = "NormalRng",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 0) throw std::invalid_argument("loading NormalRng: incorrect number of arguments");
      auto lval = FlowLoader::lvalue(is, ':');
      if (lval != "m") throw std::invalid_argument("loading NormalRng: expected mu");
      float m = FlowLoader::oneFloat(is);

      lval = FlowLoader::lvalue(is, ':');
      if (lval != "sd") throw std::invalid_argument("loading NormalRng: expected sigma");
      float sd = FlowLoader::oneFloat(is);

      lval = FlowLoader::lvalue(is, ':');
      if (lval != "seed") throw std::invalid_argument("loading NormalRng: expected sigma");
      uint64_t seed = FlowLoader::oneUint64(is);

      return new Normal(m, sd, seed);
    }
  };
  return &nl;
}

void Normal::save(std::ostream& os) const {
  os << "\n  ";
  FlowSaver::assignment(os, "m", ": ");
  FlowSaver::oneFloat(os, m_);
  os << "\n  ";
  FlowSaver::assignment(os, "sd", ": ");
  FlowSaver::oneFloat(os, sd_);
  os << "\n  ";
  FlowSaver::assignment(os, "seed", ": ");
  FlowSaver::oneFloat(os, seed_);
}

*/

}
}
}
