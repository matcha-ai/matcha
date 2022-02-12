#include "bits_of_matcha/fn/exp.h"

#include <matcha/engine>


namespace matcha {
namespace fn {

Tensor exp(const Tensor& a) {
  auto* node = new engine::fn::Exp(a);
  auto* out  = new engine::Tensor(node->out(0));
  return Tensor::fromObject(out);
}

}
}


namespace matcha {
namespace engine {
namespace fn {

Exp::Exp(Tensor* a)
  : Fn{a}
{
  wrapComputation("Exp", {in(0)});
  deduceStatus();
}

Exp::Exp(const matcha::Tensor& a)
  : Exp(deref(a))
{}

/*

const NodeLoader* Exp::getLoader() const {
  return loader();
}

const NodeLoader* Exp::loader() {
  static NodeLoader nl = {
    .type = "Exp",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 1) throw std::invalid_argument("loading Exp: incorrect number of arguments");
      return new Exp(ins[0]);
    }
  };
  return &nl;
};

*/

}
}
}
