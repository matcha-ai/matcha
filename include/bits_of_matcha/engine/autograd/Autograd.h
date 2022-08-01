#pragma once

#include "bits_of_matcha/engine/op/Op.h"

namespace matcha::engine {
class Module;
class Chain;
}

namespace matcha::engine {

struct Autograd final : Op {
  explicit Autograd(std::shared_ptr<Module> module, const std::vector<Tensor*>& wrt);
  explicit Autograd(Chain graph, const std::vector<Tensor*>& wrt);
  static OpMeta<Autograd> meta;

  void run() override;

private:
  std::shared_ptr<Module> module_;
  std::vector<Tensor*> wrt_;
};

}