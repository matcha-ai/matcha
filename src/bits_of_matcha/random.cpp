#include "bits_of_matcha/random.h"
#include "bits_of_matcha/engine/tensor/factories.h"
#include "bits_of_matcha/engine/ops/Uniform.h"
#include "bits_of_matcha/engine/ops/Normal.h"

#include <random>

using namespace matcha::engine;


namespace matcha {

random::Uniform uniform;
random::Normal normal;

}

namespace matcha::random {

Generator Uniform::init() {
  return (Generator) [this] (auto shape) mutable {
    auto op = new engine::ops::Uniform {
      deref(a),
      deref(b),
      shape,
      (size_t) rand()
    };
    auto out = ref(op->outputs[0]);
    engine::send(op);
    return out;
  };
}

Generator Normal::init() {
  return (Generator) [this] (auto shape) mutable {
    auto op = new engine::ops::Normal {
      deref(m),
      deref(sd),
      shape,
      (size_t) rand()
    };
    auto out = ref(op->outputs[0]);
    engine::send(op);
    return out;
  };
}

}