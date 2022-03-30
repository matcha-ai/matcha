#include "bits_of_matcha/engine/flow/Tracer.h"
#include "bits_of_matcha/engine/flow/Flow.h"
#include "bits_of_matcha/engine/Tensor.h"
#include "bits_of_matcha/engine/Node.h"
#include "bits_of_matcha/print.h"


namespace matcha::engine {


Tracer* Tracer::current_ = nullptr;

Tracer::Tracer()
  : open_{false}
{}

Tracer::~Tracer() {
  if (open_) current_ = nullptr;
}

Tuple Tracer::open(const std::vector<Frame>& ins) {
  Tuple tuple;
  tuple.reserve(ins.size());

  for (auto& frame: ins) {
    auto t = new Tensor(frame);
    graph_.ins.push_back(t);
    tuple.push_back(tensor(t));
  }

  if (current_) throw std::runtime_error("already tracing");
  open_ = true;
  current_ = this;
  return tuple;
}

void Tracer::close(const Tuple& outs) {
  current_ = nullptr;
  open_ = false;

  for (auto& out: outs) {
    auto t = deref(out);
    graph_.outs.push_back(t);
  }
  for (auto* node: graph_.nodes) {
//    print(node);
//    print(node->degIn(), " -> ", node->degOut());
  }
}

void Tracer::add(Node* node) {
  graph_.nodes.push_back(node);
}

void Tracer::add(Tensor* tensor) {
  graph_.tensors.push_back(tensor);
}

Tracer* Tracer::current() {
  return current_;
}

Flow* Tracer::collect() {
  if (graph_.ins.empty()) throw std::runtime_error("no Flow to collect");
  graph_.initCtx();
  return new Flow(graph_);
}

}