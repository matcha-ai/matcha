#include "bits_of_matcha/grad.h"
#include "bits_of_matcha/ops.h"
#include "bits_of_matcha/Backprop.h"


namespace matcha {

fn grad(const fn& function, const std::vector<int>& argnum) {
  nary_fn f = [function, argnum] (tuple inputs) {
    std::vector<tensor*> wrt;
    std::transform(argnum.begin(), argnum.end(),
                   std::back_inserter(wrt),
                   [&](int i) { return &inputs[i]; });

    Backprop backprop(wrt);

    tuple y = function(inputs);

    if (y.size() != 1)
      throw std::runtime_error("differentiated function must have exactly one output");

    tuple result;
    for (auto&& [t, g]: backprop(y[0]))
      result.push_back(g);

    return result;
  };
  return f;
}

fn value_and_grad(const fn& function, const std::vector<int>& argnum) {
  nary_fn f = [function, argnum] (tuple inputs) {
    std::vector<tensor*> wrt;
    std::transform(argnum.begin(), argnum.end(),
                   std::back_inserter(wrt),
                   [&](int i) { return &inputs[i]; });

    Backprop backprop(wrt);
    tuple result = function(inputs);
    if (result.size() != 1)
      throw std::runtime_error("differentiated function must have exactly one output");

    for (auto&& [t, g]: backprop(result[0]))
      result.push_back(g);

    return result;
  };
  return f;
}


}