#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/flow/graph/Graph.h"
#include "bits_of_matcha/engine/flow/Tasks.h"


namespace matcha::engine {

class Module : public Op {
public:
  Module(const std::vector<Tensor*>& inputs, const Graph& graph);
  static OpMeta<Module> meta;

  void init() override;
  void run() override;

private:
  Graph graph_;
  Tasks tasks_;

  friend class ModuleBack;
};

}