#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/flow/graph/Graph.h"
#include "bits_of_matcha/print.h"

#include <vector>
#include <algorithm>


namespace matcha::engine {

class Graph;

template <class Value>
class OpDict {
public:
  explicit OpDict(Graph* graph, Value defaultValue = {})
    : graph_(graph)
    , values_(new Value[size()])
  {
    std::fill(begin(), end(), defaultValue);
  }

  explicit OpDict(Graph& graph, Value defaultValue = {})
    : OpDict(&graph, defaultValue)
  {}

  ~OpDict() {
    delete[] values_;
  }

  OpDict(const OpDict& dict)
    : graph_(dict.graph_)
    , values_(new Value[size()])
  {
    std::copy(dict.begin(), dict.end(), begin());
  }

  OpDict& operator=(const OpDict& dict) {
    std::copy(dict.begin(), dict.end(), begin());
    return *this;
  }

  const Value& operator[](Op* op) const {
    int key = op->ctx().key();
    if (key < 0 || key >= size()) throw std::out_of_range("key is out of range");
    return values_[key];
  }

  Value& operator[](Op* op) {
    int key = op->ctx().key();
    if (key < 0 || key >= size()) throw std::out_of_range("key is out of range");
    return values_[key];
  }

  const Value* begin() const { return values_; }
  const Value* end() const { return begin() + size(); }
  Value* begin() { return values_; }
  Value* end() { return begin() + size(); }
  size_t size() const { return graph_->ops.size(); }

  bool operator==(const OpDict& other) const {
    return std::equal(begin(), end(), other.begin());
  }

  bool operator!=(const OpDict& other) const {
    return !operator==(other);
  }

protected:
  Graph* graph_;
  Value* values_;
};

}
