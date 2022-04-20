#include "bits_of_matcha/random.h"
#include "bits_of_matcha/engine/tensor/factories.h"

#include <random>

using namespace matcha::engine;


namespace matcha {

random::Uniform uniform;
random::Normal normal;

}

namespace matcha::random {

Generator Uniform::init() {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dis(a, b);

  return [=] (auto shape) mutable {
    auto tensor = generate([&]() { return dis(gen); }, shape);
    return ref(tensor);
  };
}

Generator Normal::init() {
  std::mt19937 gen(seed);
  std::normal_distribution<float> dis(m, sd);

  return [=] (auto shape) mutable {
    auto tensor = generate([&]() { return dis(gen); }, shape);
    return ref(tensor);
  };
}

}