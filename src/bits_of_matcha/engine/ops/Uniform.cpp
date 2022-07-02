#include "bits_of_matcha/engine/ops/Uniform.h"


namespace matcha::engine::ops {

Uniform::Uniform(Tensor* a, Tensor* b, const Shape& shape, size_t seed)
  : Op{a, b}
  , gen_(seed)
{
  if (a->rank() != 0 || b->rank() != 0) {
    throw std::invalid_argument("Uniform parameters must be scalars");
  }
  outputs.add(this, Float, shape);
}

OpMeta<Uniform> Uniform::meta {
  .name = "Uniform"
};

void Uniform::run() {
  auto a = *inputs[0]->buffer().as<float*>();
  auto b = *inputs[1]->buffer().as<float*>();
  std::uniform_real_distribution<float> dis(a, b);
  auto buff = outputs[0]->malloc().as<float*>();
  for (size_t i = 0; i < outputs[0]->size(); i++) {
    buff[i] = dis(gen_);
  }
}

}