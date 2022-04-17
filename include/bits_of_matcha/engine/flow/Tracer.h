#pragma once

#include "bits_of_matcha/ops.h"
#include "bits_of_matcha/Frame.h"
#include "bits_of_matcha/engine/flow/graph/Graph.h"

#include <set>

namespace matcha::engine {

class Tensor;
class Op;

class Tracer {
public:
  static bool handleOp(Op* op);
  static bool handleNewTensor(Tensor* tensor);
  static bool handleOldTensor(Tensor* tensor);

public:
  static Tracer* current();
  Tracer();

  tuple open(const std::vector<Frame>& frames);
  void close(const tuple& outputs);
  Graph collect();

private:
  Graph graph_;
  std::set<Tensor*> tensors_;
  std::set<Op*> ops_;

private:
  static Tracer* current_;

};

Graph trace(const AnyOp& op, const std::vector<Frame>& frames);

}