#include "bits_of_matcha/nn/SGD.h"


namespace matcha::nn {

SGD::operator Solver() {
  /*
   *  solver needs
   *  - dataset (to set batch size e.g.)
   *  - access to parameters, to update them
   *  - forward function, to plug in dataset instances
   */


  return [epochs = epochs, loss = loss](const UnaryFn& forward, const std::vector<tensor*>& params, const Dataset& dataset) {
//    /*
    auto batch_forw = [&](const Tuple& batch) {
      tensor data = batch[0];
      tensor gold = batch[1];
      tensor pred = forward(data);
      tensor l = loss(pred, gold);
      tensor a = pred == gold;
      return Tuple{l, a};
    };

    auto foo = matcha::flow(batch_forw);
    foo.grad.add(params);

    for (auto batch: dataset) {
//      auto loss_pred = foo({batch["x"]});
      for (auto& [param, delta]: foo.grad()) {
        *param -= 1e-4 * delta;
      }
    }
//     */
  };
}

}