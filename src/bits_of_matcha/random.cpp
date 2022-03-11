#include "bits_of_matcha/random.h"
#include "bits_of_matcha/engine/tensor.h"
#include "bits_of_matcha/print.h"

#include <random>


namespace matcha {

Random::Random(std::function<engine::Tensor*(const Shape&)> generator)
  : generator_(generator)
{}

Tensor Random::operator()(const Shape& shape) {
  auto out = generator_(shape);
  return Tensor::fromOut(out);
}

}

namespace matcha::random {


Random random = Uniform{};

Normal::operator Random() const {
  std::mt19937 gen{seed};
  std::normal_distribution<float> dist(m, sd);

  auto generator = [gen, dist](const Shape& shape) mutable {
    auto tensor = new engine::Tensor(Float, shape);
    auto buffer = tensor->writeBuffer();
    auto values = (float*) buffer->payload();

    for (size_t i = 0; i < tensor->size(); i++) {
      values[i] = dist(gen);
    }

    return tensor;
  };

  return Random(generator);
}

Tensor Normal::operator()(const Shape& shape) {
  return zz_internal_(shape);
}

Uniform::operator Random() const {
  std::mt19937 gen{seed};
  std::uniform_real_distribution<float> dist(a, b);

  auto generator = [gen, dist](const Shape& shape) mutable {
    auto tensor = new engine::Tensor(Float, shape);
    auto buffer = tensor->writeBuffer(CPU);
    auto values = (float*) buffer->payload();

    for (size_t i = 0; i < tensor->size(); i++) {
      values[i] = dist(gen);
    }

    return tensor;
  };

  return Random(generator);
}

Tensor Uniform::operator()(const Shape& shape) {
  return zz_internal_(shape);
}


}