#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/fn/transpose.h"
#include "bits_of_matcha/fn/reshape.h"
#include "bits_of_matcha/plt.h"

#include <matcha/engine>
#include <iostream>
#include <cmath>


namespace matcha {


Tensor::Tensor(const Dtype& dtype, const Shape& shape)
  : Object(new engine::Tensor(dtype, shape))
{}

Tensor::Tensor(const Input& input)
  : Object(new engine::Tensor(input.object()->out()))
{}

Tensor::Tensor(const Params& params)
  : Object(new engine::Tensor(params.object()->out()))
{}

Tensor::Tensor(float scalar)
  : Tensor(Input(scalar))
{}

Tensor::Tensor(const std::vector<float>& content)
  : Tensor(Input(content))
{}

Tensor::Tensor(const std::vector<std::vector<float>>& content)
  : Tensor(Input(content))
{}

Tensor::Tensor(const std::vector<std::vector<std::vector<float>>>& content)
  : Tensor(Input(content))
{}

Tensor::Tensor(const std::vector<std::vector<std::vector<std::vector<float>>>>& content)
  : Tensor(Input(content))
{}

Tensor::Tensor(std::initializer_list<float> content)
  : Tensor(Input(content))
{}

Tensor::Tensor(std::initializer_list<std::vector<float>> content)
  : Tensor(Input(content))
{}

Tensor::Tensor(std::initializer_list<std::vector<std::vector<float>>> content)
  : Tensor(Input(content))
{}

Tensor::Tensor(std::initializer_list<std::vector<std::vector<std::vector<float>>>> content)
  : Tensor(Input(content))
{}

const Dtype& Tensor::dtype() const {
  if (isNull()) throw std::runtime_error("Object is null");
  return object()->dtype();
}

const Shape& Tensor::shape() const {
  if (isNull()) throw std::runtime_error("Object is null");
  return object()->shape();
}

size_t Tensor::rank() const {
  return shape().rank();
}

size_t Tensor::size() const {
  return shape().size();
}

void Tensor::use(const Device& device) const {
  if (isNull()) throw std::runtime_error("Object is null");
}

void Tensor::update() const {
  if (isNull()) throw std::runtime_error("Object is null");
  object()->updateStatusChanged();
}

Tensor Tensor::reshape(const Shape& shape) const {
  return fn::reshape(*this, shape);
}

Tensor Tensor::transpose() const {
  return fn::transpose(*this);
}

Tensor Tensor::t() const {
  return fn::transpose(*this);
}

Tensor& Tensor::subst(const Tensor& source) {
  if (isNull() || source.isNull()) throw std::runtime_error("Object is null");
  object()->subst(source.object()->out());
  return *this;
}

Tensor& Tensor::subst(const Stream& source) {
  if (isNull() || source.isNull()) throw std::runtime_error("Object is null");
  object()->subst();
  source.object()->open(-1, object());
  return *this;
}

Tensor& Tensor::subst() {
  if (isNull()) throw std::runtime_error("Object is null");
  object()->subst();
  return *this;
}

void* Tensor::data() const {
  if (isNull()) throw std::runtime_error("Object is null");
  return object()->data();
}

Plt Tensor::plt() const {
  return Plt(*this);
}

Tensor Tensor::fromObject(engine::Tensor* object) {
  return Tensor(object, 0);
}

Tensor::Tensor(engine::Tensor* object, char dummy)
  : Object(object)
{}

engine::Tensor* Tensor::object() const {
  return reinterpret_cast<engine::Tensor*>(Object::object());
}

Tensor floats(const Shape& shape) {
  return Tensor(Dtype::Float, shape);
}

std::ostream& operator<<(std::ostream& os, const matcha::Tensor& tensor) {
  using namespace matcha;

  if (tensor.isNull()) throw std::invalid_argument("Object is null");
  engine::Tensor* object = tensor.object();

  if (object->status().data) {
    if (object->dtype() == Dtype::Float) {
      auto* data = (float*)(object->data());

      os << "Tensor ";
      engine::FlowSaver::flatFloats(os, data, tensor.shape());
      os << std::endl;

    } else {
      throw std::runtime_error("unkonwn dtype");
    }
  } else {
    os << tensor.dtype() << tensor.shape();
    os << std::endl;
  }

  return os;
}

}
