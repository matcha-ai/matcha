#pragma once

#include "bits_of_matcha/tensor.h"


namespace matcha::engine {

class Flow;

class FlowTracer {
public:
  FlowTracer();
  ~FlowTracer();

  Tuple open(const std::vector<Frame>& ins);
  void close(const Tuple& outs);

  Flow* get();


  static FlowTracer* current();
  void add(Node* node);

private:
  std::vector<Tensor*> ins_;
  std::vector<Tensor*> outs_;
  std::vector<Node*> nodes_;
  bool open_;

private:
  static FlowTracer* current_;

};

}
