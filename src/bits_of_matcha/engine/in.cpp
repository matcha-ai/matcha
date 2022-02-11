#include "bits_of_matcha/engine/in.h"
#include "bits_of_matcha/engine/out.h"
#include "bits_of_matcha/engine/object.h"


namespace matcha {
namespace engine {


In::In(Out* source, Object* target, unsigned id)
  : source_{source}
  , target_{target}
  , id_{id}
{
  source_->bind(this);
}

In::~In() {
  source_->unbind(this);
}

unsigned In::id() const {
  return id_;
}

void In::eval() {
  source_->eval();
}

void In::dataStatusChanged() {
  target_->dataStatusChanged(this);
}

void In::updateStatusChanged() {
  target_->updateStatusChanged(this);
}

const Dtype& In::dtype() const {
  return source_->dtype();
}

const Shape& In::shape() const {
  return source_->shape();
}

const Status& In::status() const {
  return source_->status();
}

size_t In::rank() const {
  return shape().rank();
}

size_t In::size() const {
  return shape().size();
}

device::Buffer* In::buffer() {
  return source_->buffer();
}

void In::bufferChanged() {
  target_->bufferChanged(this);
}

}
}
