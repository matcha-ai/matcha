#include "bits_of_matcha/engine/node.h"
#include "bits_of_matcha/engine/tensor.h"
#include "bits_of_matcha/tensor.h"

#include <stdexcept>
#include <algorithm>


namespace matcha {
namespace engine {

Node::Node(std::initializer_list<Tensor*> ins)
  : required_{true}
  , ins_{ins}
  , ready_{false}
{
  for (auto& in: ins) in->bindOut(this);
  ready_ = checkReady();
}

void Node::addIn(const Dtype &dtype, const Shape &shape) {
  ins_.push_back(new Tensor(dtype, shape));
  ins_.back()->bindOut(this);
}

void Node::addOut(const Dtype& dtype, const Shape& shape) {
  outs_.push_back(new Tensor(dtype, shape));
  outs_.back()->bindIn(this, outs_.size() - 1);
}

void Node::addIn(Tensor* tensor) {
  ins_.push_back(tensor);
  tensor->bindOut(this);
}

void Node::addOut(Tensor* tensor) {
  outs_.push_back(tensor);
  tensor->bindIn(this, outs_.size() - 1);
}

void Node::removeIn(Tensor* tensor) {
  auto position = std::find(
    std::begin(ins_), std::end(ins_),
    tensor
  );
  ins_.erase(position);
}

void Node::removeOut(Tensor* tensor) {
  auto position = tensor->edgeId();
  outs_.erase(std::begin(outs_) + position);
}

bool Node::required() const {
  return required_;
}

void Node::evalIns() const {
  for (auto& in: ins_) in->eval();
}

void Node::requireOuts() const {
  if (required()) return;
  required_ = true;
  for (auto& out: outs_) out->require();
}

void Node::notifyBufferChanged(Tensor* tensor) const {
  auto locateIn = std::find(
    std::begin(ins_), std::end(ins_),
    tensor
  );
  if (locateIn == std::end(ins_)) throw std::runtime_error("not an in tensor");
  int index = std::distance(std::begin(ins_), locateIn);
  onBufferChanged(index, tensor->buffer());
}

bool Node::checkReady() const {
  return std::all_of(
    std::begin(ins_), std::end(ins_),
    [=](auto* in) { return in->ready(); }
  );
}

void Node::notifyReady(Tensor* tensor, bool ready) const {
  bool newReady = checkReady();
  if (newReady == ready_) return;
  ready_ = newReady;
  for (auto* out: outs_) out->setReady(ready_);
}

void Node::onBufferChanged(int index, const device::Buffer* buffer) const {
}

Tensor* Node::out(int index) const {
  if (index < 0) index += outs();
  if (index < 0 || index >= outs()) throw std::out_of_range("node out index is out of range");
  return outs_[index];
}

size_t Node::outs() const {
  return outs_.size();
}

Tensor* Node::in(int index) const {
  if (index < 0) index += ins();
  if (index < 0 || index >= ins()) throw std::out_of_range("node in index is out of range");
  return ins_[index];
}

size_t Node::ins() const {
  return ins_.size();
}

bool Node::ready() const {
  return ready_;
}

Tensor* Node::openIn() {
  return nullptr;
}

Tensor* Node::openOut() {
  return nullptr;
}

bool Node::openIn(Tensor *tensor) {
  return false;
}

bool Node::openOut(Tensor *tensor) {
  return false;
}

bool Node::closeIn(Tensor *tensor) {
  return false;
}

bool Node::closeOut(Tensor *tensor) {
  return false;
}

bool Node::polymorphicIns() const {
  return false;
}

bool Node::polymorphicOuts() const {
  return false;
}

void Node::unrequire() const {
  required_ = false;
}

Tensor* Node::deref(const matcha::Tensor* tensor) {
  return tensor->object();
}

Tensor* Node::deref(const matcha::Tensor& tensor) {
  return tensor.object();
}

void Node::considerPruning() {
  if (referenced()) return;

  bool outReferenced = std::any_of(
    std::begin(outs_), std::end(outs_),
    [](auto* tensor) {
      return tensor->referenced();
    }
  );

  if (outReferenced) return;

  // prune node and its outs

  for (auto* in: ins_) in->unbindOut(this);
  for (auto* out: outs_) delete out;
  delete this;
}

}
}
