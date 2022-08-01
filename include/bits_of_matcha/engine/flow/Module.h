#pragma once

#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/chain/Chain.h"

#include <set>
#include <map>
#include <memory>
#include <vector>
#include <stack>


namespace matcha::engine {

class Executor;

class Module final {
public:
  explicit Module(Chain graph);
  explicit Module(const AnyOp& preimage, const std::vector<Frame>& frames);
  ~Module();

  auto chain() -> Chain&;
  auto chain() const -> const Chain&;

  void pass(const Pass& pass);

  void run(const std::vector<Tensor*>& ins, std::vector<Tensor*>& outs);
  auto run(const std::vector<Tensor*>& ins) -> std::vector<Tensor*>;

private:
  Chain chain_;
  Executor* executor_;
  Executor* executor();

};

}