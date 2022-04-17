#pragma once

#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/engine/flow/graph/Graph.h"
#include "bits_of_matcha/print.h"

#include <vector>
#include <algorithm>


namespace matcha::engine {

class Graph;

template <class Value>
class TensorDict {
public:
  explicit TensorDict(Graph* graph, Value defaultValue = {})
    : graph_(graph)
    , values_(new Value[size()])
  {
    std::fill(begin(), end(), defaultValue);
  }

  explicit TensorDict(Graph& graph, Value defaultValue = {})
    : TensorDict(&graph, defaultValue)
  {}

  ~TensorDict() {
    delete[] values_;
  }

  TensorDict(const TensorDict& dict)
    : graph_(dict.graph_)
    , values_(new Value[size()])
  {
    std::copy(dict.begin(), dict.end(), begin());
  }

  TensorDict& operator=(const TensorDict& dict) {
    std::copy(dict.begin(), dict.end(), begin());
    return *this;
  }

  const Value& operator[](Tensor* tensor) const {
    int key = tensor->ctx().key();
    if (key < 0 || key >= size()) throw std::out_of_range("key is out of range");
    return values_[key];
  }

  Value& operator[](Tensor* tensor) {
    int key = tensor->ctx().key();
    if (key < 0 || key >= size()) throw std::out_of_range("key is out of range");
    return values_[key];
  }

  const Value* begin() const {
    const auto& val = values_[0];
    return &val;
  }
  const Value* end() const { return begin() + size(); }
  Value* begin() { return values_; }
  Value* end() { return begin() + size(); }
  size_t size() const { return graph_->tensors.size(); }

  bool operator==(const TensorDict& other) const {
    return std::equal(begin(), end(), other.begin());
  }

  bool operator!=(const TensorDict& other) const {
    return !operator==(other);
  }

protected:
  Graph* graph_;
  Value* values_;
};

}
