#pragma once

#include "bits_of_matcha/nn/Loss.h"
#include "bits_of_matcha/nn/Optimizer.h"
#include "bits_of_matcha/nn/Callback.h"
#include "bits_of_matcha/nn/callbacks/Logger.h"
#include "bits_of_matcha/dataset/Dataset.h"
#include "matcha/tensor"


namespace matcha::nn {


class Net {
public:
  void fit(Dataset ds);

  tensor operator()(const tensor& a);
  tensor operator()(const tensor& a, const tensor& b);
  tensor operator()(const tensor& a, const tensor& b, const tensor& c);
  tuple operator()(const tuple& inputs);

public:
  // Sequential API

  Net(std::initializer_list<UnaryOp> sequence);
  Net(const std::vector<UnaryOp>& sequence);

public:
  // Functional API

  Net(const AnyOp& function);

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
  Optimizer optimizer;
  Loss loss;

  std::vector<std::shared_ptr<Callback>> callbacks = {
    Logger(),
  };

  size_t flops() const;
  size_t flops(const std::vector<Frame>& frames) const;
  size_t flops(const std::vector<tensor>& tensors) const;

protected:
  void initCallbacks();

  // signals

  void epochBegin(size_t epoch, size_t max);
  void epochEnd(size_t epoch, size_t max);

  void batchBegin(size_t batch, size_t max);
  void batchEnd(size_t batch, size_t max);

private:
  AnyOp function_;
  Flow trainFlow_, evalFlow_;

};


}

namespace matcha {
using nn::Net;
}