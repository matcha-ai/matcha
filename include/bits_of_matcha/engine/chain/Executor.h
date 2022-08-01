#pragma once

#include "bits_of_matcha/engine/chain/Chain.h"
#include "bits_of_matcha/engine/op/Op.h"


namespace matcha::engine {

class Executor {
public:
  explicit Executor(Chain&& chain);
  virtual ~Executor() = default;

  auto chain() -> Chain&;
  auto chain() const -> const Chain&;

  virtual auto run(const std::vector<Tensor*>& ins, std::vector<Tensor*>& outs) -> void;
  virtual auto run(const std::vector<Tensor*>& ins) -> std::vector<Tensor*>;

protected:
  virtual void runInternal() = 0;

  Chain chain_;
};

}