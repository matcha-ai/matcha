#include "bits_of_matcha/engine/lambda/passes/init.h"
#include "bits_of_matcha/engine/op/Op.h"


namespace matcha::engine {

void init(Lambda& lambda) {
  for (auto&& op: lambda.ops) if (op) op->init();
}

}