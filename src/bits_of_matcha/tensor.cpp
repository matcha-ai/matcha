#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/ops.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/engine/tensor/factories.h"

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

const Frame& tensor::frame() const {
  return deref(this)->frame();
}

const Dtype& tensor::dtype() const {
  return frame().dtype();
}

const Shape& tensor::shape() const {
  return frame().shape();
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


tensor tensor::transpose() const {
  return matcha::transpose(*this);
}

tensor tensor::t() const {
  return matcha::transpose(*this);
}

tensor tensor::dot(const tensor& b) {
  return matcha::dot(*this, b);
}


}

std::ostream& operator<<(std::ostream& os, const matcha::tensor& t) {
  deref(t)->repr(os);
  return os;
}