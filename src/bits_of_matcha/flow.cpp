#include "bits_of_matcha/flow.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/tuple.h"
#include "bits_of_matcha/engine/flow.h"

#include <fstream>


namespace matcha {


Flow::Flow(const Tensor& out)
  : Object(new engine::Flow({out}))
{}

Flow::Flow(const Tuple& outs)
  : Object(new engine::Flow(outs))
{}

Flow::Flow(std::initializer_list<Tensor> outs)
  : Object(new engine::Flow(outs))
{}

Tensor Flow::operator()(const Tensor& in) const {
}

Tuple Flow::operator()(const Tuple& ins) const {
}

void Flow::test(const Stream& stream) const {
  if (isNull()) throw std::runtime_error("object is null");
}

void Flow::save(const std::string& filepath) const {
  std::ofstream file(filepath);
  if (!file) throw std::invalid_argument("file could not be opened");
  save(file);
}

void Flow::save(std::ostream& os) const {
  if (isNull()) throw std::runtime_error("object is null");
  object()->save(os);
}

Flow Flow::load(const std::string& filepath) {
  std::ifstream file(filepath);
  if (!file) throw std::invalid_argument("file could not be opened");
  return load(file);
}

Flow Flow::load(std::istream& is) {
  return engine::Flow::load(is);
}

Flow Flow::fromObject(engine::Flow* object) {
  return Flow(object, 0);
}

Flow::Flow(engine::Flow* object, char dummy)
  : Object(object)
{}

engine::Flow* Flow::object() const {
  return reinterpret_cast<engine::Flow*>(Object::object());
}

}

std::ostream& operator<<(std::ostream& os, const matcha::Flow& flow) {
  os << "Flow { "
     << "location: " << flow.object() << "; "
     << "}" << std::endl;

  return os;
}
