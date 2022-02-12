#include "bits_of_matcha/stream.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/input.h"
#include "bits_of_matcha/engine/stream.h"
#include "bits_of_matcha/engine/input.h"
#include "bits_of_matcha/engine/nodeloader.h"

#include "bits_of_matcha/fn/batch.h"
#include "bits_of_matcha/fn/map.h"
#include "bits_of_matcha/fn/fold.h"


namespace matcha {

Stream::operator bool() const {
  if (isNull()) throw std::runtime_error("Object is null");
  return !object()->eof();
}

void Stream::reset() const {
  if (isNull()) throw std::runtime_error("Object is null");
  object()->reset();
}

void Stream::shuffle() const {
  if (isNull()) throw std::runtime_error("Object is null");
  object()->shuffle();
}

size_t Stream::size() const {
  if (isNull()) throw std::runtime_error("Object is null");
  return object()->size();
}

Stream Stream::batch(size_t sizeLimit) {
  return fn::batch(*this, sizeLimit);
}

Stream Stream::map(std::function<Tensor (const Tensor&)> fn) {
  return fn::map(*this, fn);
}

Tensor Stream::fold(const Tensor& init, std::function<Tensor (const Tensor&, const Tensor&)> fn) {
  return fn::fold(*this, init, fn);
}

Stream Stream::fromObject(engine::Stream* object) {
  return Stream(object, 0);
}

Stream::Stream(engine::Stream* object, char dummy)
  : Object(object)
{}

engine::Stream* Stream::object() const {
  return reinterpret_cast<engine::Stream*>(Object::object());
}

}

matcha::Stream& operator>>(matcha::Stream& stream, matcha::Tensor& tensor) {
  if (stream.isNull() || tensor.isNull()) throw std::runtime_error("object is null");
  tensor.object()->subst();
  stream.object()->open(tensor.object());
  return stream;
}

std::ostream& operator<<(std::ostream& os, const matcha::Stream& stream) {
//  os << stream.object()->getLoader()->type + " {/* ... */}" << std::endl;
  os << "Stream { name: " << stream.name() << " }" << std::endl;
  return os;
}
