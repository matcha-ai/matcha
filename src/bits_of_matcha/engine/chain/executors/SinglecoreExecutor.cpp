#include "bits_of_matcha/engine/chain/executors/SinglecoreExecutor.h"
#include "bits_of_matcha/engine/chain/passes/check.h"


namespace matcha::engine {

SinglecoreExecutor::SinglecoreExecutor(Chain&& chain)
  : Executor(std::move(chain))
{}

void SinglecoreExecutor::runInternal() {
//  check(chain);
  constexpr bool debug = false;

  for (auto&& op: chain_.ops) {
    if constexpr (debug) {
      std::cout << ops::name(op) << std::endl;
      for (auto&& in: op->inputs) std::cout << in << " ";
      std::cout << " ->  ";
      for (auto&& out: op->outputs) std::cout << out << " ";
      std::cout << std::endl;
    }

    op->run();
  }

  for (auto&& t: chain_.inputs) {
    if (t->op()) t->free();
  }

//  for (auto&& t: chain.tensors) {
//    if (t->op() && !t->refs()) t->free();
//  }
}

}