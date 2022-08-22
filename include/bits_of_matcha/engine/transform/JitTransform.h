#pragma once

#include "bits_of_matcha/engine/transform/TracingTransform.h"


namespace matcha::engine {

class JitTransform : public TracingTransform {
public:
  explicit JitTransform(const fn& function);
  explicit JitTransform() = default;

protected:
  std::shared_ptr<Executor> compile(Lambda lambda) override;
};

}