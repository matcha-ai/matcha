#include "bits_of_matcha/jit.h"
#include "bits_of_matcha/engine/decorator/Decorator.h"

#include <memory>


namespace matcha {

fn jit(const fn& function) {
  return ref(std::make_shared<engine::Decorator>(function));
}

}