#pragma once

#include "bits_of_matcha/nn/Loss.h"
#include "bits_of_matcha/nn/Optimizer.h"
#include "bits_of_matcha/nn/Callback.h"
#include "bits_of_matcha/nn/callbacks/Logger.h"
#include "bits_of_matcha/nn/optimizers/Sgd.h"
#include "bits_of_matcha/nn/optimizers/Adam.h"
#include "bits_of_matcha/dataset/Dataset.h"
#include "bits_of_matcha/ops.h"
#include "bits_of_matcha/fn.h"
#include "bits_of_matcha/random.h"

#include <set>

namespace matcha::nn {


class Net {
public:
  void fit(Dataset ds, size_t epochs = 10);
  void epoch(Dataset ds);
  void step(Instance i);

  tensor operator()(const tensor& a);
  tensor operator()(const tensor& a, const tensor& b);
  tensor operator()(const tensor& a, const tensor& b, const tensor& c);
  tuple operator()(const tuple& inputs);

public:
  // Sequential API

  Net(std::initializer_list<unary_fn> sequence);
  Net(const std::vector<unary_fn>& sequence);

public:
  // Functional API

  Net(const fn& forward);

  template <class Function,
  std::enable_if_t<std::is_constructible<fn, Function>::value, bool> = true>
  Net(const Function& forward) : Net((fn) forward) {}

protected:
  // Subclassing API
  Net();

  virtual void init(const tensor& a);
  virtual void init(const tensor& a, const tensor& b);
  virtual void init(const tensor& a, const tensor& b, const tensor& c);
  virtual void init(const tuple& inputs);

  virtual tensor run(const tensor& a);
  virtual tensor run(const tensor& a, const tensor& b);
  virtual tensor run(const tensor& a, const tensor& b, const tensor& c);
  virtual tuple run(const tuple& inputs);

  virtual void trainStep(Instance i);

  tensor forward(const tensor& a);
  tensor forward(const tensor& a, const tensor& b);
  tensor forward(const tensor& a, const tensor& b, const tensor& c);
  tuple forward(const tuple& inputs);

public:

  class Params {
  public:
    void add(tensor* tensor);
    void remove(const tensor* t) const;
    bool contains(const tensor* t) const;

    template <class Iterable>
    void add(const Iterable& iterable) {
      for (auto&& param: iterable) add(param);
    }

    std::_Rb_tree_const_iterator<tensor*> begin();
    std::_Rb_tree_const_iterator<tensor*> end();

    std::_Rb_tree_const_iterator<tensor*> begin() const;
    std::_Rb_tree_const_iterator<tensor*> end() const;

    size_t size() const;
    size_t total() const;

  private:
    std::set<tensor*> tensors_;
  };

  Params params;

  Optimizer optimizer = Adam{};
  Loss loss;

  std::vector<std::shared_ptr<Callback>> callbacks = {
    Logger(),
  };

protected:
  // Callback events

  void fitInit();
  void fitBegin(Dataset ds);
  void fitEnd();

  void epochBegin(size_t epoch, size_t max);
  void epochEnd();

  void batchBegin(size_t batch, size_t max);
  void batchEnd();

  void propagateForward(const Instance& instance, const tensor& loss);
  void propagateBackward(const std::map<tensor*, tensor>& gradients);

private:
  fn forward_;
  bool initialized_;

};


}

namespace matcha {
using nn::Net;
}