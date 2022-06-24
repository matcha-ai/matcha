#pragma once

#include "bits_of_matcha/tensor.h"

#include <vector>
#include <map>

namespace matcha {

/**
 * @brief Dynamic backpropagation controller
 */
class Backprop {
public:
  /**
   * @param wrt target gradient tensors
   */
  Backprop(std::initializer_list<tensor*> wrt);

  /**
   * @param wrt target gradient tensors
   */
  Backprop(const std::vector<tensor*>& wrt);

  /**
   * @param wrt target gradient tensors
   */
  template <class Wrt>
  Backprop(const Wrt& wrt)
    : Backprop(std::vector(std::begin(wrt), std::end(wrt)))
  {}

  /**
   * @param root tensor to backpropagate from
   * @return gradients w.r.t. wrt tensors
   * @see Backprop::Backprop
   */
  std::map<tensor*, tensor> operator()(const tensor& root);

  ~Backprop();

private:
  std::vector<tensor*> wrt_;
  void* interal_;
};

}