#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"

using namespace matcha::engine;

namespace matcha {

tensor::tensor()
  : internal_(new Tensor({}))
{
  std::cout <<"internal: " << internal_ << std::endl;
}

tensor::tensor(float scalar)
  : internal_(engine::full(scalar, {}))
{}

tensor& tensor::operator=(const tensor& other) {
  return *this;
}

tensor tensor::full(float value, const Shape& shape) {
  return ref(engine::full(value, shape));
}

tensor tensor::zeros(const Shape& shape) {
  return ref(engine::zeros(shape));
}

tensor tensor::ones(const Shape& shape) {
  return ref(engine::ones(shape));
}

tensor tensor::eye(const Shape& shape) {
  return ref(engine::eye(shape));
}


}

std::ostream& operator<<(std::ostream& os, const matcha::tensor& t) {
  deref(t)->repr(os);
  return os;
}