#include "bits_of_matcha/object.h"
#include "bits_of_matcha/engine/object.h"

#include <iostream>


namespace matcha {

Object::Object()
  : object_{nullptr}
{}

Object::Object(engine::Object* object)
  : Object()
{
  reset(object);
}

Object::Object(const Object& other)
  : Object()
{
  reset(other.object());
}

Object::~Object() {
  if (isNull()) return;
  this->object()->unbindRef(this);
}

engine::Object* Object::object() const {
  return object_;
}

bool Object::isNull() const {
  return object_ == nullptr;
}

void Object::reset(engine::Object* object) const {
  if (!isNull()) {
    this->object()->unbindRef(this);
  }
  object_ = object;
  if (!isNull()) {
    this->object()->bindRef(this);
  }
}

void Object::release() const {
  if (isNull()) return;
  this->object()->unbindRef(this, false);
  object_ = nullptr;
}

const Object& Object::operator=(const Object& other) {
  reset(other.object());
  return *this;
}

const std::string& Object::name() const {
  if (isNull()) throw std::runtime_error("object is null");
  return object()->name();
}

void Object::rename(const std::string& name) const {
  if (isNull()) throw std::runtime_error("object is null");
  object()->rename(name);
}


}
