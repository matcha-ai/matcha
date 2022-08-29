#include "bits_of_matcha/engine/lambda/executors/SinglecoreExecutor.h"
#include "bits_of_matcha/engine/lambda/passes/debug.h"

#include <map>


namespace matcha::engine {

SinglecoreExecutor::SinglecoreExecutor(Lambda&& lambda)
  : Executor(std::move(lambda))
{
  std::map<Tensor*, size_t> dependencies;

  for (auto&& t: lambda_.tensors) {
    if (!t->op()) continue;
    dependencies[t] = t->reqs() - 1;
  }

  for (auto&& op: lambda_.ops) {
    instructions_.emplace_back(op);

    for (auto&& in: op->inputs) {
      if (!dependencies.contains(in)) continue;
      auto& deps = dependencies[in];

      if (!deps)
        throw std::runtime_error("deps are already 0");

      deps--;

      if (!deps)
        instructions_.emplace_back(in);
    }
  }
}

void SinglecoreExecutor::runInternal() {
//  engine::debug(lambda_);
  constexpr bool debug = false;

  if constexpr (debug)
    std::cout << std::string(64, '=') << std::endl;

  for (auto&& i: instructions_) {
    if constexpr (debug) {
      if (std::holds_alternative<Op*>(i)) {
        auto&& op = std::get<Op*>(i);

        std::string opname = ops::name(op);
        std::cout << opname << " ";
        if (opname.size() < 20)
          std::cout << std::string(20 - opname.size(), ' ');

        for (auto&& in: op->inputs) std::cout << in << " ";
        if (!op->outputs.empty()) std::cout << " ->  ";
        for (auto&& out: op->outputs) std::cout << out << " ";
        std::cout << std::endl;

      } else {
        std::cout << "free " << std::get<Tensor*>(i) << std::endl;
      }

    }

    if (std::holds_alternative<Op*>(i))
      std::get<Op*>(i)->run();
    else
      std::get<Tensor*>(i)->free();
  }

  if constexpr (debug)
    std::cout << std::string(64, '=') << std::endl;
}

}