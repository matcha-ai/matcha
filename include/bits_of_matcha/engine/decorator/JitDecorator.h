#pragma once

#include "bits_of_matcha/engine/decorator/TracingDecorator.h"


namespace matcha::engine {

class JitDecorator : public TracingDecorator {
public:
  explicit JitDecorator(const fn& function);
  explicit JitDecorator() = default;

protected:
  std::shared_ptr<Executor> compile(Chain chain) override;
};

}