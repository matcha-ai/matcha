#include "bits_of_matcha/nn/Net.h"
#include "bits_of_matcha/nn/Layer.h"
#include "bits_of_matcha/Backprop.h"
#include "bits_of_matcha/decorators.h"


namespace matcha::nn {

Net::Net(const fn& function)
  : forward_(function)
{}

Net::Net(const std::vector<UnaryOp>& sequence) {
  forward_ = [sequence] (tensor feed) {
    for (auto& op: sequence) feed = op(feed);
    return feed;
  };
  forward_ = jit(forward_);
}

Net::Net(std::initializer_list<UnaryOp> sequence)
  : Net(std::vector(sequence))
{}

void Net::trainStep(Instance i) {
  tensor x = i["x"];
  tensor t = i["y"];

  Backprop backprop;
  tensor y = forward_(x);
  tensor l = loss(t, y);

  auto gradients = backprop(l, params);

  propagateForward(i, l);

  for (auto&& [param, grad]: gradients)
    optimizer(*param, grad);

  propagateBackward(gradients);
}

void Net::step(Instance i) {
  if (Layer::netStack_.empty() || Layer::netStack_.top() != this)
    Layer::netStack_.push(this);

  trainStep(std::move(i));

  if (!Layer::netStack_.empty() && Layer::netStack_.top() == this)
    Layer::netStack_.pop();
}

void Net::epoch(Dataset ds) {
  if (Layer::netStack_.empty() || Layer::netStack_.top() != this)
    Layer::netStack_.push(this);

  int batch = 0;
  for (Instance&& i: ds) {
    batchBegin(batch, ds.size());
    step(i);
    batchEnd();
    batch++;
  }

  if (!Layer::netStack_.empty() && Layer::netStack_.top() == this)
    Layer::netStack_.pop();
}

void Net::fit(Dataset ds, size_t epochs) {
  Layer::netStack_.push(this);

  fitInit();
  step(ds.get());
  ds.reset();

  fitBegin(ds);
  for (int e = 0; e < epochs; e++) {
    epochBegin(e, epochs);
    epoch(ds);
    epochEnd();
  }
  fitEnd();

  if (!Layer::netStack_.empty() && Layer::netStack_.top() == this)
    Layer::netStack_.pop();
}

void Net::fitInit() {
  for (auto&& cb: callbacks) if (cb) cb->onfitInit(*this);
}

void Net::fitBegin(Dataset ds) {
  for (auto&& cb: callbacks) if (cb) cb->onfitBegin(*this, ds);
}

void Net::fitEnd() {
  for (auto&& cb: callbacks) if (cb) cb->onfitEnd(*this);
}

void Net::epochBegin(size_t epoch, size_t max) {
  for (auto&& cb: callbacks) if (cb) cb->onEpochBegin(epoch, max);
}

void Net::epochEnd() {
  for (auto&& cb: callbacks) if (cb) cb->onEpochEnd();
}

void Net::batchBegin(size_t batch, size_t max) {
  for (auto&& cb: callbacks) if (cb) cb->onBatchBegin(batch, max);
}

void Net::batchEnd() {
  for (auto&& cb: callbacks) if (cb) cb->onBatchEnd();
}

void Net::propagateForward(const Instance& instance, const tensor& l) {
  for (auto&& cb: callbacks) if (cb) cb->onPropagateForward(instance, l);
}

void Net::propagateBackward(const std::map<tensor*, tensor>& gradients) {
  for (auto&& cb: callbacks) if (cb) cb->onPropagateBackward(gradients);
}

void Net::Params::add(tensor* tensor) {
  tensors_.insert(tensor);
}

std::_Rb_tree_const_iterator<tensor*> Net::Params::begin() {
  return tensors_.begin();
}

std::_Rb_tree_const_iterator<tensor*> Net::Params::end() {
  return tensors_.end();
}

std::_Rb_tree_const_iterator<tensor*> Net::Params::begin() const {
  return tensors_.begin();
}

std::_Rb_tree_const_iterator<tensor*> Net::Params::end() const {
  return tensors_.end();
}

size_t Net::Params::size() const {
  return tensors_.size();
}

size_t Net::Params::total() const {
  size_t total = 0;
  for (auto&& t: tensors_) total += t->size();
  return total;
}


}