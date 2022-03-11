#include "bits_of_matcha/engine/flowBuilder.h"
#include "bits_of_matcha/engine/flow.h"
#include "bits_of_matcha/print.h"


namespace matcha::engine {

FlowBuilder* FlowBuilder::current_ = nullptr;

FlowBuilder::FlowBuilder(Flow* flow)
  : flow_{flow}
{
//  print("FlowBuilder(Flow*)");
  if (current_ != nullptr) throw std::runtime_error("cannot build multiple flows at once");
  current_ = this;
}

FlowBuilder::~FlowBuilder() {
//  print("~FlowBuilder()");
  current_ = nullptr;
}

FlowBuilder* FlowBuilder::current() {
  return current_;
}

void FlowBuilder::add(Tensor* tensor) {
//  print("Building ", tensor);
  tensors_.push_back(tensor);
}

void FlowBuilder::finish() {
//  print("In total ", tensors_.size(), " tensors");
  flow_->check();
}

void FlowBuilder::inlet(std::vector<Tensor*> tensors) {
//  print(tensors.size(), " inlets");
}

void FlowBuilder::outlet(std::vector<Tensor*> tensors) {
//  print(tensors.size(), " outlets");
}

}