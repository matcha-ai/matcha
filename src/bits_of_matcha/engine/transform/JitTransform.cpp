#include "bits_of_matcha/engine/transform/JitTransform.h"
#include "bits_of_matcha/engine/lambda/passes/matmulFusion.h"
#include "bits_of_matcha/engine/lambda/passes/inlineExpansion.h"
#include "bits_of_matcha/engine/lambda/passes/deadCodeElimination.h"
#include "bits_of_matcha/engine/lambda/passes/copyPropagation.h"
#include "bits_of_matcha/engine/lambda/passes/constantPropagation.h"
#include "bits_of_matcha/engine/lambda/passes/init.h"
#include "bits_of_matcha/engine/lambda/passes/debug.h"
#include "bits_of_matcha/engine/lambda/executors/SinglecoreExecutor.h"

#include <fstream>

namespace matcha::engine {

JitTransform::JitTransform(const fn& function)
  : TracingTransform(function)
{}

std::shared_ptr<Executor> JitTransform::compile(Lambda lambda) {
//  std::ofstream before("/home/patz/temp/before.txt");
//  std::ofstream after("/home/patz/temp/after.txt");
//
//  deadCodeElimination(lambda);
//  copyPropagation(lambda);
//  debug(lambda, before);

  inlineExpansion(lambda);
  deadCodeElimination(lambda);
  copyPropagation(lambda);
  matmulFusion(lambda);
  constantPropagation(lambda);

//  debug(lambda, after);
//  debug(lambda);
  init(lambda);
  return std::make_shared<SinglecoreExecutor>(std::move(lambda));
}

}