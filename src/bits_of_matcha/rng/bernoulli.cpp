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

void Bernoulli::reset() {

}

void Bernoulli::shuffle() {

}

bool Bernoulli::eof() const {
  return false;
}

size_t Bernoulli::size() const {
  return std::numeric_limits<size_t>::max();
}

Tensor* Bernoulli::open() {
  auto* out = new Tensor(Dtype::Float, {});
  open(out);
  return out;
}

void Bernoulli::open(Tensor* tensor) {
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

/*

const NodeLoader* Bernoulli::getLoader() const {
  return loader();
}

const NodeLoader* Bernoulli::loader() {
  static NodeLoader nl = {
    .type = "BernoulliRng",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 0) throw std::invalid_argument("loading BernoulliRng: incorrect number of arguments");
      auto lval = FlowLoader::lvalue(is, ':');
      if (lval != "m") throw std::invalid_argument("loading BernoulliRng: expected mu");
      float m = FlowLoader::oneFloat(is);

      lval = FlowLoader::lvalue(is, ':');
      if (lval != "sd") throw std::invalid_argument("loading BernoulliRng: expected sigma");
      float sd = FlowLoader::oneFloat(is);

      lval = FlowLoader::lvalue(is, ':');
      if (lval != "seed") throw std::invalid_argument("loading BernoulliRng: expected sigma");
      uint64_t seed = FlowLoader::oneUint64(is);

      return new Bernoulli(m, sd, seed);
    }
  };
  return &nl;
}

void Bernoulli::save(std::ostream& os) const {
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
