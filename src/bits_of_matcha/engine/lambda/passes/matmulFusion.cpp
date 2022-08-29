#include "bits_of_matcha/engine/lambda/passes/matmulFusion.h"
#include "bits_of_matcha/engine/ops/Matmul.h"
#include "bits_of_matcha/engine/ops/Transpose.h"
#include "bits_of_matcha/engine/ops/Identity.h"


namespace matcha::engine {

void matmulFusion(Lambda& lambda) {
  std::vector<Op*> new_ops;
  std::set<Op*> orphan_ops;
  std::set<Tensor*> orphan_tensors;

  for (auto&& op: lambda.ops) {
    if (typeid(*op) != typeid(ops::Matmul)) {
      new_ops.push_back(op);
      continue;
    }

    char transpose[2];
    for (int i = 0; i < 2; i++) {
      auto in = op->inputs[i];
      auto iop = in->op();
      transpose[i] = 'N';

      if (!iop) continue; // No op, therefore no transpose.
      if (in->reqs() > 2) continue; // explicit computation required for other purposes
      if (typeid(*iop) != typeid(ops::Transpose)) continue; // not a transpose
      transpose[i] = 'T';
      orphan_ops.insert(iop);
      orphan_tensors.insert(in);
    }

    Tensor* a = transpose[0] == 'N' ? op->inputs[0] : op->inputs[0]->op()->inputs[0];
    Tensor* b = transpose[1] == 'N' ? op->inputs[1] : op->inputs[1]->op()->inputs[0];
    Tensor* c = op->outputs[0];

    delete op;
    auto fused = new ops::Matmul(a, b, {transpose[0], transpose[1]});
    new_ops.push_back(fused);
    lambda.tensors.push_back(fused->outputs[0]);
    fused->outputs[0]->req();

    auto id = new ops::Identity(fused->outputs[0], c);
    new_ops.push_back(id);
  }

  lambda.ops.clear();
  for (auto&& op: new_ops)
    if (!orphan_ops.contains(op))
      lambda.ops.push_back(op);

  std::vector<Tensor*> tensors = std::move(lambda.tensors);
  for (auto&& t: tensors) {
    if (orphan_tensors.contains(t))
      t->unreq();
    else
      lambda.tensors.push_back(t);
  }
}

}