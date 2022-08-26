#include "bits_of_matcha/nn/Net.h"
#include "bits_of_matcha/nn/Layer.h"
#include "bits_of_matcha/Backprop.h"
#include "bits_of_matcha/transforms.h"


namespace matcha::nn {

Net::Net(const fn& function)
  : initialized_(false)
{
  forward_ = function;
}

Net::Net(const std::vector<UnaryOp>& sequence)
  : initialized_(false)
{
  forward_ = [sequence] (tensor feed) {
    for (auto& op: sequence) feed = op(feed);
    return feed;
  };
  forward_ = jit(forward_);
}

Net::Net(std::initializer_list<unary_fn> sequence)
  : Net(std::vector(sequence))
{}

tensor Net::forward(const tensor& a) {
  if (!initialized_) {
    initialized_ = true;
    init(a);
  }
  return run(a);
}

tensor Net::forward(const tensor& a, const tensor& b) {
  if (!initialized_) {
    initialized_ = true;
    init(a, b);
  }
  return run(a, b);
}

tensor Net::forward(const tensor& a, const tensor& b, const tensor& c) {
  if (!initialized_) {
    initialized_ = true;
    init(a, b, c);
  }
  return run(a, b, c);
}

tuple Net::forward(const tuple& inputs) {
  if (!initialized_) {
    initialized_ = true;
    init(inputs);
  }
  return run(inputs);
}

void Net::trainStep(Instance i) {
  tensor x = i["x"];
  tensor t = i["y"];
  size_t bsize = x.shape()[0];

  Backprop backprop;
  tensor y = forward(x);
  tensor l = loss(t, y);

  auto gradients = backprop(l, params);

  propagateForward(i, l);
  for (auto&& [p, g]: gradients) g /= bsize;
  optimizer(gradients);
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

  epochBegin(0, 1);
  int batch = 0;
  for (Instance&& i: ds) {
    batchBegin(batch, ds.size());
    step(i);
    batchEnd();
    batch++;
  }
  epochEnd();

  if (!Layer::netStack_.empty() && Layer::netStack_.top() == this)
    Layer::netStack_.pop();
}

void Net::fit(Dataset ds, size_t epochs) {
  Layer::netStack_.push(this);

  fitInit();
  ds.reset();
  step(ds.get());
  ds.reset();

  fitBegin(ds);
  for (int e = 0; e < epochs; e++) {
    epochBegin(e, epochs);
    int batch = 0;
    for (Instance&& i: ds) {
      batchBegin(batch, ds.size());
      step(i);
      batchEnd();
      batch++;
    }
    epochEnd();
  }
  fitEnd();

  if (!Layer::netStack_.empty() && Layer::netStack_.top() == this)
    Layer::netStack_.pop();
}

tensor Net::operator()(const tensor& a) {
  return forward(a);
}

tensor Net::operator()(const tensor& a, const tensor& b) {
  return forward(a, b);
}

tensor Net::operator()(const tensor& a,
                       const tensor& b,
                       const tensor& c)
{
  return forward(a, b, c);
}

tuple Net::operator()(const tuple& inputs) {
  return forward(inputs);
}

Net::Net()
{}

void Net::init(const tensor& a) {}
void Net::init(const tensor& a, const tensor& b) {}
void Net::init(const tensor& a, const tensor& b, const tensor& c) {}
void Net::init(const tuple& inputs) {}

tensor Net::run(const tensor& a) {
  return forward_(a);
}

tensor Net::run(const tensor& a, const tensor& b) {
  return forward_(a, b);
}

tensor Net::run(const tensor& a, const tensor& b, const tensor& c) {
  return forward_(a, b, c);
}

tuple Net::run(const tuple& tuple) {
  return forward_(tuple);
}

void Net::fitInit() {
  for (auto&& cb: callbacks) if (cb) cb->onFitInit(*this);
}

void Net::fitBegin(Dataset ds) {
  for (auto&& cb: callbacks) if (cb) cb->onFitBegin(*this, ds);
}

void Net::fitEnd() {
  for (auto&& cb: callbacks) if (cb) cb->onFitEnd(*this);
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