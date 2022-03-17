#include "bits_of_matcha/rng.h"
#include "bits_of_matcha/Generator.h"

#include <random>


namespace matcha::rng {


Generator Uniform::init() const {
  std::mt19937 gen{seed};
  std::uniform_real_distribution<float> dist(a, b);

  auto generator = [gen, dist](const Shape& shape) mutable {
    auto t = new engine::Tensor(Float, shape);
    auto buffer = t->writeBuffer(CPU);
    auto values = (float*) buffer->payload();

    for (size_t i = 0; i < t->size(); i++) {
      values[i] = dist(gen);
    }

    return tensor(t);
  };

  return Generator(generator);
}

Generator Normal::init() const {
  std::mt19937 gen{seed};
  std::normal_distribution<float> dist(m, sd);

  auto generator = [gen, dist](const Shape& shape) mutable {
    auto t = new engine::Tensor(Float, shape);
    auto buffer = t->writeBuffer(CPU);
    auto values = (float*) buffer->payload();

    for (size_t i = 0; i < t->size(); i++) {
      values[i] = dist(gen);
    }

    return tensor(t);
  };

  return Generator(generator);
}

}
