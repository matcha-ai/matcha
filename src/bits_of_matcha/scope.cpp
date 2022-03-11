#pragma once

#include "bits_of_matcha/scope.h"
#include "bits_of_matcha/print.h"

#include <stdexcept>


namespace matcha {

Scope::Scope() {
  scopes_.push(this);
}

Scope::Scope(const std::string& name) {

}

Scope::~Scope() {
//  if (scopes_.top() != this); std::cerr << "scope stack corruption" << std::endl;
//  scopes_.pop();
}

}