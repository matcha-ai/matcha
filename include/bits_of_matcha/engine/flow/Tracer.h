#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/flow/Graph.h"


namespace matcha::engine {

class Flow;

class Tracer {
public:
  Tracer();
  ~Tracer();

  Tuple open(const std::vector<Frame>& ins);
  void close(const Tuple& outs);

  Flow* collect();


  static Tracer* current();
  void add(Node* node);
  void add(Tensor* tensor);

private:
  Graph graph_;
  bool open_;

private:
  static Tracer* current_;

};

}
