#include "bits_of_matcha/engine/object.h"
#include "bits_of_matcha/object.h"
#include "bits_of_matcha/engine/in.h"
#include "bits_of_matcha/engine/out.h"

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

In* Object::createIn(Out* source, unsigned id) {
  return new In(source, this, id);
}

Out* Object::createOut(const Dtype& dtype, const Shape& shape, unsigned id) {
  return new Out(dtype, shape, this, id);
}

void Object::release(const matcha::Object* object) {
  object->release();
}

void Object::release(const matcha::Object& object) {
  object.release();
}

void Object::bindRef(const matcha::Object* ref) {
//  std::cout << "binding ref" << std::endl;
  refCount_++;
}

void Object::unbindRef(const matcha::Object *ref, bool autoPrune) {
  if (!referenced()) throw std::runtime_error("Object is not referenced");
  refCount_--;
  if (!autoPrune) return;
  if (!referenced()) prune();
}

const std::string& Object::name() const {
  return name_;
}

void Object::rename(const std::string& name) const {
  name_ = name;
}

const Status& Object::status() const {
  return status_;
}

void Object::dataStatusChanged(In* in) {

}

void Object::updateStatusChanged(In* in) {

}

void Object::bufferChanged(In* in) {

}

void Object::eval(Out* out) {

}

}
}
