#include "bits_of_matcha/engine/ops/Normal.h"


namespace matcha::engine::ops {

Normal::Normal(Tensor* m, Tensor* sd, const Shape& shape, size_t seed)
  : Op{m, sd}
  , gen_(seed)
{
  if (m->rank() != 0 || sd->rank() != 0) {
    throw std::invalid_argument("Normal parameters must be scalars");
  }
  outputs.add(this, Float, shape);
}

OpMeta<Normal> Normal::meta {
  .name = "Normal"
};

void Normal::run() {
  auto m = *inputs[0]->buffer().as<float*>();
  auto sd = *inputs[1]->buffer().as<float*>();
  std::normal_distribution<float> dis(m, sd);
  auto buff = outputs[0]->malloc().as<float*>();
  for (size_t i = 0; i < outputs[0]->size(); i++) {
    buff[i] = dis(gen_);
  }
}

}
