#include "bits_of_matcha/engine/flow/FlowTracer.h"
#include "bits_of_matcha/engine/Tensor.h"
#include "bits_of_matcha/engine/Node.h"
#include "bits_of_matcha/print.h"


namespace matcha::engine {


FlowTracer* FlowTracer::current_ = nullptr;

FlowTracer::FlowTracer()
  : open_{false}
{
  print("created Tracker");
}

FlowTracer::~FlowTracer() {
  if (open_) current_ = nullptr;
}

Tuple FlowTracer::open(const std::vector<Frame>& ins) {
  Tuple tuple;
  tuple.reserve(ins.size());

  for (auto& frame: ins) {
    auto t = new Tensor(frame);
    ins_.push_back(t);
    tuple.push_back(tensor(t));
  }

  if (current_) throw std::runtime_error("already tracing");
  open_ = true;
  current_ = this;
  return tuple;
}

void FlowTracer::close(const Tuple& outs) {
  current_ = nullptr;
  open_ = false;

  for (auto& out: outs) {
    auto t = deref(out);
    outs_.push_back(t);
  }
  for (auto* node: nodes_) {
    print(node);
    print(node->ins(), " -> ", node->outs());
  }
}

void FlowTracer::add(Node* node) {
  nodes_.push_back(node);
}

FlowTracer* FlowTracer::current() {
  return current_;
}

Flow* FlowTracer::get() {
  return nullptr;
}

}