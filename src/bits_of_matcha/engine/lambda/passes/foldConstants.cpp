#include "bits_of_matcha/engine/lambda/passes/constantPropagation.h"


namespace matcha::engine {

void constantPropagation(Lambda& lambda) {
  std::set<Tensor*> non_constant;

  for (auto&& in: lambda.inputs)
    non_constant.insert(in);

  for (auto&& [in, binding]: lambda.side_inputs)
    non_constant.insert(in);

  std::vector<Op*> ops;
  for (auto&& op: lambda.ops) {

  }
}

}