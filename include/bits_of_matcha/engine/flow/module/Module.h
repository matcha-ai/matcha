#pragma once

#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/flow/Graph.h"
#include "bits_of_matcha/engine/flow/Tasks.h"
#include "bits_of_matcha/engine/flow/Backprop.h"

#include <set>
#include <map>
#include <memory>
#include <vector>
#include <stack>


namespace matcha::engine {

class Graph;
class Tasks;
class ModuleForw;

class Module {
public:
  explicit Module(std::unique_ptr<Graph>&& graph);

  std::vector<Frame> framesIn() const;
  std::vector<Frame> framesOut() const;

  std::vector<Tensor*> forward(const std::vector<Tensor*>& ins);

  using Partial = std::pair<Tensor*, std::vector<Tensor*>>;
  using Partials = std::map<Tensor*, Partial>;

  void forward(
    const std::vector<Tensor*>& inputs,
    std::vector<Tensor*>& outputs
  );

  std::vector<Tensor*> forward(
    const std::vector<Tensor*>& inputs,
    Partials& partials,
    const std::vector<Tensor*>& wrt = {}
  );

  void forward(
    const std::vector<Tensor*>& inputs,
    std::vector<Tensor*>& outputs,
    Partials& partials,
    const std::vector<Tensor*>& wrt = {}
  );

  void backward(
    Partials& partials,
    const std::vector<Tensor*>& deltas = {}
  );

  static void accumulateGrads(Partial& partial, const Shape& shape);

private:
  std::unique_ptr<Graph> graph_;
  std::stack<Backprop> backprops_;

  static void stream(const std::vector<Tensor*>& source, std::vector<Tensor*>& target);
  static std::vector<Tensor*> collect(const std::vector<Tensor*>& source);

  friend class ModuleForw;
  friend class ModuleBack;

private:
  void forwardRegularOp(Op* op, Partials& partials);
  void forwardModuleOp(ModuleForw* op, Partials& partials);
  void backwardRegularOp(Op* op, Partials& partials);
  void backwardModuleOp(ModuleForw* op, Partials& partials);

  void schedulePartials(std::map<Tensor*, Partial>& partials,
                        const std::vector<Tensor*>& wrt);

  void plugDeltas(const std::vector<Tensor*>& deltas,
                  std::map<Tensor*, Partial>& partials);

  Op* prepareOpBack(Op* op, std::map<Tensor*, Partial>& partials);

};

}