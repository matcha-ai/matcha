#pragma once

#include "bits_of_matcha/ops.h"
#include "bits_of_matcha/engine/flow/Graph.h"

#include <stack>
#include <vector>
#include <set>
#include <map>
#include <memory>

namespace matcha::engine {

class Graph;

std::unique_ptr<Graph> trace(const AnyOp& op, const std::vector<Frame>& frames);
bool tracing();

void incept(Op* op, Op* preop);

class Tracer {
  friend std::unique_ptr<Graph> trace(const AnyOp&, const std::vector<Frame>&);
  friend bool tracing();
  friend void incept(Op*, Op*);

  thread_local static std::stack<Tracer*> tracings_;
  static bool active();
  static Tracer* get();

  std::unique_ptr<Graph> graph_;
  std::set<Tensor*> addedTensors_;

public:
  Tracer();
  ~Tracer();

  tuple open(const std::vector<Frame>& frames);
  std::unique_ptr<Graph> close(const tuple& outputs);

private:
  bool addNewOp(Op* op);
  bool addNewTensor(Tensor* tensor);
  bool addOldTensor(Tensor* tensor);

public:
  static bool handleNewOp(Op* op);
  static bool handleNewTensor(Tensor* tensor);
  static bool handleOldTensor(Tensor* tensor);
};

}