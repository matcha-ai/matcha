#pragma once

#include "bits_of_matcha/engine/flow/graph/TensorDict.h"


namespace matcha::engine {

class TensorMask : public TensorDict<bool> {
public:
  explicit TensorMask(Graph* graph, bool defaultValue = false);
  explicit TensorMask(Graph& graph, bool defaultValue = false);

  TensorMask operator~() const;
  TensorMask operator&(const TensorMask& mask) const;
  TensorMask operator|(const TensorMask& mask) const;
  TensorMask& operator&=(const TensorMask& mask);
  TensorMask& operator|=(const TensorMask& mask);

  size_t count() const;
  std::vector<Tensor*> get() const;
  std::vector<Tensor*> rget() const;
};

}