#include "bits_of_matcha/engine/inlet.h"


namespace matcha {
namespace engine {


Inlet::Inlet(Tensor* input)
  : Node{}
{
  addOut(input);
}

bool Inlet::openIn(Tensor *tensor) {
  return false;
}

Tensor* Inlet::openIn() {
  return nullptr;
}

const NodeLoader* Inlet::loader() {
  return nullptr;
}

const NodeLoader* Inlet::getLoader() const{
  return loader();
}

void Inlet::save(std::ostream& os) const {

}

bool Inlet::polymorphicIns() const {
  return true;
}

}
}
