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

Tensor Stream::operator()() {
  return Tensor::fromObject(object()->open(-1));
}

Tensor Stream::operator()(int idx) {
  if (idx < 0) throw std::invalid_argument("idx < 0");
  return Tensor::fromObject(object()->open(idx));
}

void Stream::reset() const {
  if (isNull()) throw std::runtime_error("Object is null");
  object()->seek(0);
}

void Stream::shuffle() const {
  if (isNull()) throw std::runtime_error("Object is null");
}

bool Stream::next() {
  if (isNull()) throw std::runtime_error("Object is null");
  return object()->next();
}

bool Stream::seek(size_t pos) {
  if (isNull()) throw std::runtime_error("Object is null");
  return object()->seek(pos);
}

size_t Stream::tell() const {
  if (isNull()) throw std::runtime_error("Object is null");
  return object()->tell();
}

size_t Stream::size() const {
  if (isNull()) throw std::runtime_error("Object is null");
  return object()->size();
}

bool Stream::eof() const {
  if (isNull()) throw std::runtime_error("Object is null");
  return object()->eof();
}

Stream::operator bool() const {
  return !eof();
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
  stream.object()->open(-1, tensor.object());
  return stream;
}

std::ostream& operator<<(std::ostream& os, const matcha::Stream& stream) {
//  os << stream.object()->getLoader()->type + " {/* ... */}" << std::endl;
  os << "Stream { name: " << stream.name() << " }" << std::endl;
  return os;
}
