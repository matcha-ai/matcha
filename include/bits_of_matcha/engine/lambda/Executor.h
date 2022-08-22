#pragma once

#include "bits_of_matcha/engine/lambda/Lambda.h"
#include "bits_of_matcha/engine/op/Op.h"


namespace matcha::engine {

class Executor {
public:
  explicit Executor(Lambda&& lambda);
  virtual ~Executor() = default;

  auto lambda() -> Lambda&;
  auto lambda() const -> const Lambda&;

  virtual auto run(const std::vector<Tensor*>& ins, std::vector<Tensor*>& outs) -> void;
  virtual auto run(const std::vector<Tensor*>& ins) -> std::vector<Tensor*>;

protected:
  virtual void runInternal() = 0;

  Lambda lambda_;
};

}