#include "bits_of_matcha/fn/sqrt.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/tuple.h"
#include "bits_of_matcha/engine/tensor.h"

#include <matcha/device>


namespace matcha {
namespace fn {

Tensor sqrt(const Tensor& a) {
  auto* node = new engine::fn::Sqrt(a);
  auto* out  = new engine::Tensor(node->out(0));
  return Tensor::fromObject(out);
}

}
}


namespace matcha {
namespace engine {
namespace fn {

Sqrt::Sqrt(Tensor* a)
  : Fn{a}
{
  wrapComputation("Sqrt", {in(0)});
  deduceStatus();
}

Sqrt::Sqrt(const matcha::Tensor& a)
  : Sqrt(deref(a))
{}

/*

const NodeLoader* Sqrt::getLoader() const {
  return loader();
}

const NodeLoader* Sqrt::loader() {
  static NodeLoader nl = {
    .type = "Sqrt",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 1) throw std::invalid_argument("loading Sqrt: incorrect number of arguments");
      return new Sqrt(ins[0]);
    }
  };
  return &nl;
};

*/

}
}
}
