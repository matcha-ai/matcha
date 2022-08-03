#include "bits_of_matcha/engine/decorator/JitDecorator.h"
#include "bits_of_matcha/engine/chain/passes/flatten.h"
#include "bits_of_matcha/engine/chain/passes/reduceToEffects.h"
#include "bits_of_matcha/engine/chain/passes/contractIdentities.h"
#include "bits_of_matcha/engine/chain/passes/initialize.h"
#include "bits_of_matcha/engine/chain/passes/debug.h"
#include "bits_of_matcha/engine/chain/executors/SinglecoreExecutor.h"


namespace matcha::engine {

JitDecorator::JitDecorator(const fn& function)
  : TracingDecorator(function)
{}

std::shared_ptr<Executor> JitDecorator::compile(Chain chain) {

  flatten(chain);
  reduceToEffects(chain);
  contractIdentities(chain);
  initialize(chain);

  return std::make_shared<SinglecoreExecutor>(std::move(chain));
}

}