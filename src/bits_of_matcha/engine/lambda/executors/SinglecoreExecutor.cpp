#include "bits_of_matcha/engine/lambda/executors/SinglecoreExecutor.h"
#include "bits_of_matcha/engine/lambda/passes/debug.h"


namespace matcha::engine {

SinglecoreExecutor::SinglecoreExecutor(Lambda&& lambda)
  : Executor(std::move(lambda))
{}

void SinglecoreExecutor::runInternal() {
//  engine::debug(lambda_);
  constexpr bool debug = false;

  if constexpr (debug)
    std::cout << std::string(64, '=') << std::endl;

  for (auto&& op: lambda_.ops) {
    if constexpr (debug) {
      std::string opname = ops::name(op);
      std::cout << opname << " ";

      if (opname.size() < 20)
        std::cout << std::string(20 - opname.size(), ' ');

      for (auto&& in: op->inputs) std::cout << in << " ";
      if (!op->outputs.empty()) std::cout << " ->  ";
      for (auto&& out: op->outputs) std::cout << out << " ";
      std::cout << std::endl;
    }

    op->run();
  }

//  for (auto&& t: lambda_.inputs) {
//    if (t->op()) t->free();
//  }

//  for (auto&& t: lambda.tensors) {
//    if (t->op() && !t->refs()) t->free();
//  }

  if constexpr (debug)
    std::cout << std::string(64, '=') << std::endl;
}

}