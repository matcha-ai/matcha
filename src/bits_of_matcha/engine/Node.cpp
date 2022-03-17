#include "bits_of_matcha/engine/Node.h"
#include "bits_of_matcha/engine/Tensor.h"
#include "bits_of_matcha/engine/memory.h"
#include "bits_of_matcha/engine/flow/FlowTracer.h"
#include "bits_of_matcha/Device.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/print.h"

#include <numeric>
#include <limits>


namespace matcha::engine {


Node::Node()
{}

Node::Node(std::initializer_list<Tensor*> ins)
  : ins_{ins}
{
  for (auto in: ins_) in->req();
  auto tracer = FlowTracer::current();
  if (tracer) tracer->add(this);
  flow_ = tracer;
}

Node::~Node() {
//  print("deleting node");
  for (auto in: ins_) in->unreq();
  for (auto out: outs_) out->setSource(nullptr);
}

Tensor* Node::in(int idx) {
  return ins_[idx];
}

Tensor* Node::out(int idx) {
  return outs_[idx];
}

int Node::ins() const {
  return (int)ins_.size();
}

int Node::outs() const {
  return (int)outs_.size();
}

int Node::inIdx(Tensor* in) const {
  return (int) std::distance(
    std::begin(ins_),
    std::find(std::begin(ins_), std::end(ins_), in)
  );
}

int Node::outIdx(Tensor* out) const {
  return (int) std::distance(
    std::begin(outs_),
    std::find(std::begin(outs_), std::end(outs_), out)
  );
}

void Node::createOut(const Frame& frame) {
  auto* out = new Tensor(frame);
  out->setSource(this);
  outs_.push_back(out);
  if (outs() > 1) throw std::runtime_error("TODO multiple-out nodes not implemented");
}

void Node::createOut(const Dtype& dtype, const Shape& shape) {
  createOut(Frame(dtype, shape));
}

void Node::init() {
  x_.resize(ins());
  y_.resize(outs());

  for (int i = 0; i < ins(); i++) {
    auto in = ins_[i];
    if (in->source()) {
      in->source()->init();
    }
    if (in->uses(device())) {
      x_[i] = in->buffer();
    } else {
      throw std::runtime_error("TODO node buffer transfers");
      x_[i] = malloc(in->bytes(), *device());
    }
  }

  for (int i = 0; i < outs(); i++) {
    auto out = outs_[i];
    out->writeBuffer(*device());
    y_[i] = out->buffer();
  }
}

void Node::run() {
  for (auto in: ins_) {
    if (in->source()) in->source()->run();
  }
}

const Device::Concrete* Node::device() const {
  return ins_[0]->device();
}

void Node::use(const Device& device) {
}

bool Node::flow() {
  return flow_;
}

}