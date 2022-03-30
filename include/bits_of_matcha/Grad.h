#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/fn.h"

#include <vector>
#include <tuple>
#include <functional>
#include <map>


namespace matcha {

class Flow;

class Grad {
public:

  void add(tensor* wrt);
  void add(std::vector<tensor*> wrt);
  void remove(std::vector<tensor*> wrt);
  void remove(tensor* wrt);

  std::vector<std::tuple<tensor*, tensor>> operator()();

  Grad& operator=(const Grad& grad) = delete;

private:
  Grad(Flow* flow);
  Flow* flow_;

  friend class Flow;

};

}