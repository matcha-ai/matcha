#pragma once


#include "bits_of_matcha/Flow.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/flow/graph/Graph.h"
#include "bits_of_matcha/engine/flow/Tasks.h"

#include <vector>
#include <set>


namespace matcha::engine {

class Flow {
public:
  explicit Flow();
  explicit Flow(const AnyOp& op);
  void setOp(const AnyOp& op);
  bool hasOp() const;

  bool built() const;
  void build(const std::vector<Frame>& frames);

  std::vector<Tensor*> forward(const std::vector<Tensor*>& inputs);

  void requireGrad(Tensor* wrt);
  void requireGrad(const std::vector<Tensor*>& wrts);
  void unrequireGrad(Tensor* wrt);
  void unrequireGrad(const std::vector<Tensor*>& wrts);
  const std::set<Tensor*>& requiredGrad();
  void setRequiredGrad(const std::vector<Tensor*>& wrts);

  TensorMask getGradMask();
  void compile();
  void compileJit();
  void updateTasks();

private:
  bool hasOp_;
  AnyOp op_;
  Graph graph_;
  Tasks tasks_;

  std::set<Tensor*> grads_;
  bool compile_;
};

matcha::Flow ref(Flow* internal);
Flow* deref(const matcha::Flow& external);
Flow* deref(const matcha::Flow* external);

}