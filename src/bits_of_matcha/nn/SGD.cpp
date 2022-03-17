#include "bits_of_matcha/nn/SGD.h"


namespace matcha::nn {

SGD::operator Solver() {
  /*
  return [epochs = epochs, loss = loss](Flow& flow, const Dataset& dataset) {
    Grad grad = flow.grad();

    for (int e = 0; e < epochs; e++) {
      for (auto i: dataset) {
        tensor data = i["x"];
        tensor gold = i["y"];
        tensor pred = flow(data);
        tensor pred_loss = loss(pred, gold);

        for (auto& [param, delta]: grad.propagate(pred_loss)) {
        }
      }
    }
  };
   */
}

}