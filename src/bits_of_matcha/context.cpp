#include "bits_of_matcha/context.h"

#include <stdexcept>
#include <iostream>


namespace matcha {

Context::Context() {
  auto* parent = parentContext();
  if (parent != nullptr) {
    device_ = parent->device_;
  }

  pushContext();
}

Context::~Context() {
  popContext();
}

Context::Context(const Device& device)
  : Context()
{
  use(device);
}

void Context::use(const Device& device) {
  device_ = &device;
}

const Device& Context::device() {
  return *parentContext()->device_;
}

Context* Context::parentContext() {
  if (contextStack_.empty()) {
    return &defaultContext_;
  } else {
    return contextStack_.top();
  }
}

void Context::pushContext() {
  contextStack_.push(this);
}

void Context::popContext() {
  if (contextStack_.top() != this) return;
  contextStack_.pop();
}

thread_local std::stack<Context*> Context::contextStack_ {};
Context Context::defaultContext_ {};

}
