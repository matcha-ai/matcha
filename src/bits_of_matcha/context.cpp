#include "bits_of_matcha/context.h"

#include <stdexcept>
#include <iostream>


namespace matcha {

Context::Context() {
  auto* parent = parentContext();
  if (parent != nullptr) {
    device_ = parent->device_;
    debug_  = parent->debug_;
  }

  pushContext();
}

Context::Context(const std::string& name)
  : Context()
{
  rename(name);
}

Context::~Context() {
  popContext();
}

const std::string& Context::name() const {
  return name_;
}

void Context::rename(const std::string& name) {
  name_ = name;
}

void Context::use(Device& device) {
  device_ = &device;
}

void Context::debug(int level) {
  debug_ = level;
}

const Context* Context::current() {
  auto& stack = contextStack_;
  if (stack.empty()) {
    return &defaultContext_;
  } else {
    return stack.top();
  }
}

const Device* Context::getDevice() const {
  return device_;
}

int Context::getDebug() const {
  return debug_;
}

Context* Context::parentContext() {
  auto& stack = contextStack_;
  if (stack.empty()) {
    return &defaultContext_;
  } else {
    return stack.top();
  }
}

void Context::pushContext() {
  contextStack_.push(this);
}

void Context::popContext() {
  if (contextStack_.top() != this) return;
  contextStack_.pop();
}

Context::Context(Device& device, int debugLevel)
  : Context()
{
  use(device);
  debug(debugLevel);
}

thread_local std::stack<Context*> Context::contextStack_ {};
Context Context::defaultContext_ {*new device::Cpu(), 0};

}
