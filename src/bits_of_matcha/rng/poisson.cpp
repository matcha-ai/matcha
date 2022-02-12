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
{

}

void Poisson::reset() {

}

void Poisson::shuffle() {

}

bool Poisson::eof() const {
  return false;
}

size_t Poisson::size() const {
  return std::numeric_limits<size_t>::max();
}

Tensor* Poisson::open() {
  auto* out = new Tensor(Dtype::Float, {});
  open(out);
  return out;
}

void Poisson::open(Tensor* tensor) {
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

/*

const NodeLoader* Poisson::getLoader() const {
  return loader();
}

const NodeLoader* Poisson::loader() {
  static NodeLoader nl = {
    .type = "PoissonRng",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 0) throw std::invalid_argument("loading PoissonRng: incorrect number of arguments");
      auto lval = FlowLoader::lvalue(is, ':');
      if (lval != "m") throw std::invalid_argument("loading PoissonRng: expected mu");
      float m = FlowLoader::oneFloat(is);

      lval = FlowLoader::lvalue(is, ':');
      if (lval != "sd") throw std::invalid_argument("loading PoissonRng: expected sigma");
      float sd = FlowLoader::oneFloat(is);

      lval = FlowLoader::lvalue(is, ':');
      if (lval != "seed") throw std::invalid_argument("loading PoissonRng: expected sigma");
      uint64_t seed = FlowLoader::oneUint64(is);

      return new Poisson(m, sd, seed);
    }
  };
  return &nl;
}

void Poisson::save(std::ostream& os) const {
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
