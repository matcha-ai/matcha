#include "bits_of_matcha/engine/transform/JitTransform.h"
#include "bits_of_matcha/engine/lambda/passes/matmulFusion.h"
#include "bits_of_matcha/engine/lambda/passes/inlineExpansion.h"
#include "bits_of_matcha/engine/lambda/passes/deadCodeElimination.h"
#include "bits_of_matcha/engine/lambda/passes/copyPropagation.h"
#include "bits_of_matcha/engine/lambda/passes/constantPropagation.h"
#include "bits_of_matcha/engine/lambda/passes/init.h"
#include "bits_of_matcha/engine/lambda/passes/debug.h"
#include "bits_of_matcha/engine/lambda/passes/check.h"
#include "bits_of_matcha/engine/lambda/executors/SinglecoreExecutor.h"

#include <fstream>

namespace matcha::engine {

JitTransform::JitTransform(const fn& function)
  : CachingTransform(function)
{}

std::shared_ptr<Executor> JitTransform::compile(Lambda lambda) {
//  std::ofstream file("/home/patz/asdf.txt");
//  debug(lambda, file);

  inlineExpansion(lambda);
  deadCodeElimination(lambda);
  matmulFusion(lambda);
  copyPropagation(lambda);
  constantPropagation(lambda);


  init(lambda);
  return std::make_shared<SinglecoreExecutor>(std::move(lambda));
}

}