#include "bits_of_matcha/transforms.h"
#include "bits_of_matcha/engine/transform/JitTransform.h"

#include <memory>


namespace matcha {

fn jit(const fn& function) {
  std::shared_ptr<engine::Transform> internal {new engine::JitTransform(function)};
  return ref(internal);
}

}