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

void Uniform::reset() {

}

void Uniform::shuffle() {

}

bool Uniform::eof() const {
  return false;
}

size_t Uniform::size() const {
  return std::numeric_limits<size_t>::max();
}

Tensor* Uniform::open() {
  auto* out = new Tensor(Dtype::Float, {});
  open(out);
  return out;
}

void Uniform::open(Tensor* tensor) {
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

/*

const NodeLoader* Uniform::getLoader() const {
  return loader();
}

const NodeLoader* Uniform::loader() {
  static NodeLoader nl = {
    .type = "UniformRng",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 0) throw std::invalid_argument("loading UniformRng: incorrect number of arguments");
      auto lval = FlowLoader::lvalue(is, ':');
      if (lval != "m") throw std::invalid_argument("loading UniformRng: expected mu");
      float m = FlowLoader::oneFloat(is);

      lval = FlowLoader::lvalue(is, ':');
      if (lval != "sd") throw std::invalid_argument("loading UniformRng: expected sigma");
      float sd = FlowLoader::oneFloat(is);

      lval = FlowLoader::lvalue(is, ':');
      if (lval != "seed") throw std::invalid_argument("loading UniformRng: expected sigma");
      uint64_t seed = FlowLoader::oneUint64(is);

      return new Uniform(m, sd, seed);
    }
  };
  return &nl;
}

void Uniform::save(std::ostream& os) const {
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
