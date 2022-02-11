#include "bits_of_matcha/fn/lnot.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/tuple.h"
#include "bits_of_matcha/engine/tensor.h"

#include <matcha/device>


namespace matcha {
namespace fn {

Tensor lnot(const Tensor& a) {
  auto* node = new engine::fn::Lnot(a);
  auto* out  = new engine::Tensor(node->out(0));
  return Tensor::fromObject(out);
}

}
}


matcha::Tensor operator!(const matcha::Tensor& a) {
  return matcha::fn::lnot(a);
}


namespace matcha {
namespace engine {
namespace fn {

Lnot::Lnot(Tensor* a)
  : Fn{a}
{
  wrapComputation("Lnot", {in(0)});
  deduceStatus();
}

Lnot::Lnot(const matcha::Tensor& a)
  : Lnot(deref(a))
{}

/*

const NodeLoader* Lnot::getLoader() const {
  return loader();
}

const NodeLoader* Lnot::loader() {
  static NodeLoader nl = {
    .type = "Lnot",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 1) throw std::invalid_argument("loading Lnot: incorrect number of arguments");
      return new Lnot(ins[0]);
    }
  };
  return &nl;
};

*/

}
}
}
