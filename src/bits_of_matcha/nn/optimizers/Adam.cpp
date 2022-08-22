#include "bits_of_matcha/nn/optimizers/Adam.h"


namespace matcha::nn {

void Adam::operator()(std::map<tensor*, tensor>& gradients) {
  internal_.t_ += 1;
  auto& ss = internal_.s_;
  auto& rs = internal_.r_;

  for (auto&& [t, g]: gradients) {
    if (!ss.contains(t)) ss[t] = zeros(t->shape());
    if (!rs.contains(t)) rs[t] = zeros(t->shape());

    auto& s = ss[t];
    auto& r = rs[t];

    s = beta1 * s + (1 - beta1) * g;
    r = beta2 * r + (1 - beta2) * square(g);

    tensor s_bar = s / (1 - power(beta1, t));
    tensor r_bar = r / (1 - power(beta2, t));
    *t -= lr * s_bar / sqrt(r_bar + epsilon);
  }

}

}