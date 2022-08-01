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
}

Net::Net(std::initializer_list<UnaryOp> sequence)
  : Net(std::vector(sequence))
{}

void Net::fit(Dataset ds, size_t epochs) {
  Layer::netStack_.push(this);
//  forward_ = jit(forward_);
//  trainFlow_ = function_;
//  trainFlow_.build({ds.get()["x"]});
  ds.reset();

  trainBegin(ds);

  for (int epoch = 0; epoch < epochs; epoch++) {
    epochBegin(epoch, epochs);
    int batch = 0;
    for (auto i: ds) {
      batchBegin(batch, ds.size());
      tensor x = i["x"];
      tensor t = i["y"];

      Backprop backprop(params);
//      print(88 * (x.reshape(-1, 28, 28) != 0), "\n\n");
//      print("---");
      tensor y = forward_(x);
//      print(argmax(y, -1).reshape(1, -1));
      tensor l = loss(t, y);

      auto gradients = backprop(l);

      propagateForward(i, l);
      for (auto&& [param, grad]: gradients) {
        optimizer(*param, grad);
      }
      propagateBackward(gradients);

      batchEnd();
      batch++;
    }
    epochEnd();
  }
  trainEnd();
  if (Layer::netStack_.top() != this)
    throw std::runtime_error("net stack corruption");
  Layer::netStack_.pop();
}

void Net::trainBegin(Dataset ds) {
  for (auto&& cb: callbacks) if (cb) cb->onTrainBegin(*this, ds);
}

void Net::trainEnd() {
  for (auto&& cb: callbacks) if (cb) cb->onTrainEnd();
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