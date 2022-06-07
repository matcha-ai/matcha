#pragma once


#include "bits_of_matcha/Flow.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/flow/graph/Graph.h"
#include "bits_of_matcha/engine/flow/Tasks.h"

#include <vector>
#include <map>
#include <set>


namespace matcha::engine {

class Flow {
public:
  explicit Flow();
  explicit Flow(const AnyOp& op);
  void setOp(const AnyOp& op);
  bool hasOp() const;

  bool submoduling() const;
  std::vector<Tensor*> submodule(const std::vector<Tensor*>& inputs);

  bool built() const;
  void build(const std::vector<Frame>& frames);

  std::vector<Tensor*> forward(const std::vector<Tensor*>& inputs);
  std::map<tensor*, tensor> backward(Tensor* delta);
  std::vector<Tensor*> call(const std::vector<Tensor*>& inputs);

  bool wantsOp() const;
  bool wantsBuild() const;

  void requireGrad(tensor* wrt);
  void requireGrad(const std::vector<tensor*>& wrts);
  void unrequireGrad(tensor* wrt);
  void unrequireGrad(const std::vector<tensor*>& wrts);
  std::set<tensor*> requiredGrad();
  void setRequiredGrad(const std::vector<tensor*>& wrts);

  TensorMask getGradMask();
  void compile();
  void compileJit();
  void updateTasks();

private:
  bool hasOp_;
  AnyOp op_;
  Graph graph_;
  Tasks tasks_;

  std::map<tensor*, Tensor*> grads_;
  bool compile_;
};

matcha::Flow ref(Flow* internal);
Flow* deref(const matcha::Flow& external);
Flow* deref(const matcha::Flow* external);

}