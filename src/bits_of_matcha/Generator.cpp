#include "bits_of_matcha/Generator.h"

namespace matcha {

Generator::Generator(const std::function<tensor(const Shape&)> generator)
: generator_{generator}
{}

tensor Generator::operator()(const Shape& shape) const {
  return generator_(shape);
}

}
