#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/params.h"
#include "bits_of_matcha/stream.h"
#include "bits_of_matcha/input.h"
#include "bits_of_matcha/engine/tensor.h"
#include "bits_of_matcha/engine/params.h"
#include "bits_of_matcha/engine/stream.h"
#include "bits_of_matcha/engine/input.h"
#include "bits_of_matcha/engine/flowsaver.h"

#include <iostream>


namespace matcha {


Tensor::Tensor(const Dtype& dtype, const Shape& shape)
  : Object(new engine::Tensor(dtype, shape))
{}

Tensor::Tensor(const Input& input)
  : Object(new engine::Tensor(input.object()->out()))
{}

Tensor::Tensor(const Stream& stream)
  : Object(stream.object()->open())
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

void Tensor::subst(const Tensor& source) {
  if (isNull() || source.isNull()) throw std::runtime_error("Object is null");
  object()->subst(source.object()->out());
}

void Tensor::subst() {
  if (isNull()) throw std::runtime_error("Object is null");
  object()->subst();
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

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
  if (tensor.isNull()) throw std::invalid_argument("Object is null");

  if (tensor.object()->status().data) {
    tensor.object()->eval();
    if (tensor.dtype() == Dtype::Float) {
      auto* data = reinterpret_cast<const float*>(tensor.object()->getData());

      os << "Tensor ";
      engine::FlowSaver::flatFloats(os, data, tensor.shape());
      os << std::endl;

    } else {
      throw std::runtime_error("unkonwn dtype");
    }
  } else {
    auto& shape = tensor.shape();
    auto& dtype = tensor.dtype();

    os << tensor.dtype() << tensor.shape();
    os << std::endl;
  }
  return os;
}

}
