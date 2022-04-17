#pragma once


#include "bits_of_matcha/Flow.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/flow/graph/Graph.h"
#include "bits_of_matcha/engine/flow/Tasks.h"

#include <vector>


namespace matcha::engine {

class Flow {
public:
  explicit Flow();
  explicit Flow(const AnyOp& op);
  void setOp(const AnyOp& op);
  bool hasOp() const;

  bool built() const;
  void build(const std::vector<Frame>& frames);

private:
  bool hasOp_;
  AnyOp op_;
  Graph graph_;
  Tasks tasks_;
};

matcha::Flow ref(Flow* internal);
Flow* deref(const matcha::Flow& external);
Flow* deref(const matcha::Flow* external);

}