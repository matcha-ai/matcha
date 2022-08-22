#include "bits_of_matcha/engine/transform/JitTransform.h"
#include "bits_of_matcha/engine/lambda/passes/inlineExpansion.h"
#include "bits_of_matcha/engine/lambda/passes/deadCodeElimination.h"
#include "bits_of_matcha/engine/lambda/passes/copyPropagation.h"
#include "bits_of_matcha/engine/lambda/passes/constantPropagation.h"
#include "bits_of_matcha/engine/lambda/passes/init.h"
#include "bits_of_matcha/engine/lambda/passes/debug.h"
#include "bits_of_matcha/engine/lambda/executors/SinglecoreExecutor.h"


namespace matcha::engine {

JitTransform::JitTransform(const fn& function)
  : TracingTransform(function)
{}

std::shared_ptr<Executor> JitTransform::compile(Lambda lambda) {

  inlineExpansion(lambda);
  deadCodeElimination(lambda);
  copyPropagation(lambda);
  constantPropagation(lambda);

//  debug(lambda);
  init(lambda);
  return std::make_shared<SinglecoreExecutor>(std::move(lambda));
}

}