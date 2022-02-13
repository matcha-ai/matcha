#include "bits_of_matcha/engine/out.h"
#include "bits_of_matcha/engine/in.h"
#include "bits_of_matcha/engine/object.h"


namespace matcha {
namespace engine {

Out::Out(const Dtype& dtype, const Shape& shape, Object* source, unsigned id)
  : dtype_{dtype}
  , shape_{shape}
  , source_{source}
  , id_{id}
  , buffer_{nullptr}
{
}

unsigned Out::id() const {
  return id_;
}

void Out::setId(unsigned id) {
  id_ = id;
}

bool Out::linked() const {
  return !targets_.empty();
}

void Out::dataStatusChanged() {
  for (auto* target: targets_) {
    target->dataStatusChanged();
  }
}

void Out::updateStatusChanged() {
  for (auto* target: targets_) {
    target->updateStatusChanged();
  }
}

void Out::bind(In* target) {
  targets_.insert(target);
}

void Out::unbind(In* target) {
  targets_.erase(target);
  if (targets_.empty()) {
    source_->prune(this);
  }
}

void Out::eval() {
  source_->eval(this);
}

const Dtype& Out::dtype() const {
  return dtype_;
}

const Shape& Out::shape() const {
  return shape_;
}

size_t Out::rank() const {
  return shape().rank();
}

size_t Out::size() const {
  return shape().size();
}

const Status& Out::status() const {
  return source_->status();
}

device::Buffer* Out::buffer() {
  return buffer_;
}

void Out::setBuffer(device::Buffer* buffer) {
  buffer_ = buffer;
  for (auto* target: targets_) target->bufferChanged();
}

}
}
