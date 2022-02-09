#include "bits_of_matcha/engine/object.h"

#include <stdexcept>
#include <iostream>
#include <string>


namespace matcha {
namespace engine {

Object::Object()
  : Object("obj" + std::to_string((uint64_t)this))
{}

Object::Object(const std::string& name)
  : refCount_{0}
  , name_{name}
{}

Object::~Object() {
  if (referenced()) {
    std::cout << "WARNING: deleted a referenced object" << std::endl;
  }
}

bool Object::referenced() const {
  return refCount_ > 0;
}

void Object::bindRef(const matcha::Object* ref) {
//  std::cout << "binding ref" << std::endl;
  refCount_++;
}

void Object::unbindRef(const matcha::Object *ref) {
//  std::cout << refCount_ << std::endl;
  if (!referenced()) throw std::runtime_error("Object is not referenced");
  refCount_--;
  if (!referenced()) considerPruning();
}

const std::string& Object::name() const {
  return name_;
}

void Object::rename(const std::string& name) const {
  name_ = name;
}


}
}
