#pragma once

#include "bits_of_matcha/fn.h"
#include "bits_of_matcha/nn/Solver.h"
#include "bits_of_matcha/data/Dataset.h"
#include "bits_of_matcha/Flow.h"

#include <initializer_list>
#include <vector>



namespace matcha::nn {

class Net {
public:
  Net(const UnaryFn& function);
  Net(std::initializer_list<UnaryFn> sequential);

  void fit(const Dataset& dataset);
  Solver solver;

  tensor operator()(const tensor& data);

  class Ctx {
  public:
    template <class... T>
    static void train(T... t);

    static bool training();
  };


private:
  UnaryFn function_;

  struct Context {
  public:
    Context();
    static const Context* current();

    bool training;

  private:
    static Context* current_;
  };
  static const Context* ctx();
  friend class Layer;

};

}

namespace matcha {
using nn::Net;
}