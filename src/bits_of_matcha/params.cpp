#include "bits_of_matcha/params.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/input.h"
#include "bits_of_matcha/stream.h"
#include "bits_of_matcha/engine/params.h"

#include <stdexcept>


namespace matcha {


Params::Params(const Dtype& dtype, const Shape& shape)
  : Object(new engine::Params(dtype, shape))
{}

Params::Params(const Dtype& dtype, const Shape& shape, const Tensor& init)
  : Object(new engine::Params(dtype, shape, init.object()))
{}

Params::Params(const Dtype& dtype, const Shape& shape, const Stream& init)
  : Object(new engine::Params(dtype, shape, init.object()))
{}

Params::Params(const Tensor& tensor)
  : Object(new engine::Params(tensor.object()))
{}

Params::Params(const Input& init)
  : Object(new engine::Params(init.object()))
{}

Params::Params(const Stream& init)
  : Object(new engine::Params(init.object()))
{}

const Params& Params::operator=(const Tensor& tensor) {
  std::cout << "assignment" << std::endl;
  return *this;
}

const Dtype& Params::dtype() const {
  if (isNull()) throw std::runtime_error("object is null");
  return object()->dtype();
}

const Shape& Params::shape() const {
  return object()->shape();
}

size_t Params::rank() const {
  return shape().rank();
}

size_t Params::size() const {
  return shape().size();
}

void Params::update(const Tensor& value) {
  if (isNull() || value.isNull()) throw std::runtime_error("object is null");
  object()->update(value.object());
}

Params Params::fromObject(engine::Params* object) {
  return Params(object, 0);
}

Params::Params(engine::Params* object, char dummy)
  : Object(object)
{}

engine::Params* Params::object() const {
  return reinterpret_cast<engine::Params*>(Object::object());
}

}

std::ostream& operator<<(std::ostream& os, const matcha::Params& params) {
  using namespace matcha;
  auto& shape = params.shape();
  auto& dtype = params.dtype();

  os << "Params { "
     << "dtype: " << dtype.string() << ", "
     << "shape: [";

  for (int i = 0; i < shape.rank(); i++) {
    if (i != 0) os << ", ";
    os << shape[i];
  }
  os << "], ";
  os << "size: " << shape.size() << " ";

  os << "}" << std::endl;
  return os;
}
