#pragma once

#include "bits_of_matcha/ops.h"
#include "bits_of_matcha/fn.h"
#include "bits_of_matcha/Flow.h"


namespace matcha {

using Decorator = std::function<fn(const fn&)>;

template <class Callable, std::enable_if_t<
std::is_constructible<fn, Callable>::value, bool> = true>
auto decorate(const Callable& callable, const Decorator& decorator) {
  fn decorated = decorator(callable);
  if constexpr (std::is_same<)
}


}