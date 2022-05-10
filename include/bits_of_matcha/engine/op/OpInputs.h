#pragma once

#include <vector>


namespace matcha::engine {

class Tensor;

class OpInputs {
public:
  OpInputs(const std::vector<Tensor*>& inputs);

  Tensor*& operator[](int idx);
  size_t size() const;

  Tensor** begin();
  Tensor** end();

  bool any() const;
  bool none() const;

  ~OpInputs();

private:
  std::vector<Tensor*> data_;
};


}
