#include "bits_of_matcha/decorators.h"
#include "bits_of_matcha/engine/decorator/JitDecorator.h"

#include <memory>


namespace matcha {

fn jit(const fn& function) {
  std::shared_ptr<engine::Decorator> dec {new engine::JitDecorator(function)};
  return ref(std::move(dec));
}

}