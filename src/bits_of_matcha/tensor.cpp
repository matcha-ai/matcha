#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/ops.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/engine/tensor/factories.h"
#include "bits_of_matcha/print.h"
#include "bits_of_matcha/engine/ops/Print.h"


using namespace matcha::engine;

namespace matcha {

tensor::tensor()
  : internal_(new Tensor({}))
{
//  std::cout <<"internal: " << internal_ << std::endl;
  deref(this)->ref();
}

tensor::tensor(float scalar)
  : internal_(engine::full(scalar, {}))
{
  deref(this)->ref();
}

tensor& tensor::operator=(const tensor& other) {
  auto temp = identity(other);
  if (internal_) deref(this)->unref();
  internal_ = unref(temp);
  return *this;
}

tensor::tensor(const tensor& other) {
  auto temp = identity(other);
  internal_ = unref(temp);
}

tensor::tensor(tensor&& other) noexcept {
  internal_ = other.internal_;
  other.internal_ = nullptr;
}

tensor::tensor(void* engineObject) {
  internal_ = engineObject;
  if (internal_) deref(this)->ref();
}

tensor::~tensor() {
  if (internal_) deref(this)->unref();
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
  auto op = new ops::Print(deref(t), false, os);
  collect(op);
  return os;
}