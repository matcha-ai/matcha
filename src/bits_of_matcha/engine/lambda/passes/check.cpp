#include "bits_of_matcha/engine/lambda/passes/check.h"
#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"


namespace matcha::engine {

bool checkOpToposort(const Lambda& lambda);
bool checkOpUniqueness(const Lambda& lambda);
bool checkTensorInclusion(const Lambda& lambda);
bool checkTensorUniqueness(const Lambda& lambda);


int check(const Lambda& lambda) {
  if (!checkOpToposort(lambda)) return 1;
  if (!checkOpUniqueness(lambda)) return 2;
  if (!checkTensorInclusion(lambda)) return 3;
  if (!checkTensorUniqueness(lambda)) return 4;

  return 0;
}


bool checkOpToposort(const Lambda& lambda) {
  std::set<Op*> visited;
  for (auto&& op: lambda.ops) {
    for (auto&& in: lambda.inputs) {
      if (in && in->op() && !visited.contains(in->op()))
        return false;
    }
  }
  return true;
}

bool checkOpUniqueness(const Lambda& lambda) {
  auto unique = std::set(lambda.ops.begin(), lambda.ops.end());
  return unique.size() == lambda.ops.size();
}

bool checkTensorUniqueness(const Lambda& lambda) {
  auto unique = std::set(lambda.tensors.begin(), lambda.tensors.end());
  return unique.size() == lambda.tensors.size();
}

bool checkTensorInclusion(const Lambda& lambda) {
  auto tensors = std::set(lambda.tensors.begin(), lambda.tensors.end());
  for (auto&& op: lambda.ops) {
    for (auto&& t: op->inputs)
      if (!tensors.contains(t)) return false;
    for (auto&& t: op->outputs)
      if (!tensors.contains(t)) return false;
  }
  for (auto&& t: lambda.inputs)
    if (!tensors.contains(t)) return false;
  for (auto&& t: lambda.outputs)
    if (!tensors.contains(t)) return false;
  for (auto&& [t, source]: lambda.side_inputs)
    if (!tensors.contains(t)) return false;

  return true;
}


}