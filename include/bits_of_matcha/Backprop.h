#pragma once

#include "bits_of_matcha/tensor.h"

#include <vector>
#include <map>

#define MATCHA_BACKPROP_API alignas(void*)

namespace matcha {

/**
 * @brief Back-propagation controller
 */
class MATCHA_BACKPROP_API Backprop {
public:
  explicit Backprop(std::initializer_list<tensor*> wrt);
  explicit Backprop(const std::vector<tensor*>& wrt);

  template <class Iterable>
  explicit Backprop(const Iterable& wrt)
    : Backprop(std::vector<tensor*>(std::begin(wrt), std::end(wrt)))
  {}

  std::map<tensor*, tensor> operator()(const tensor& root);

  ~Backprop();

private:
  void* internal_;
};

}