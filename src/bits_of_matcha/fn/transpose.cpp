#include "bits_of_matcha/fn/transpose.h"

#include <matcha/engine>


namespace matcha {
namespace fn {

Tensor transpose(const Tensor& a) {
  auto* node = new engine::fn::Transpose(a);
  auto* out  = new engine::Tensor(node->out(0));
  return Tensor::fromObject(out);
}

}
}


namespace matcha {
namespace engine {
namespace fn {

Transpose::Transpose(Tensor* a)
  : Fn{a}
{
  auto& shapeA = in(0)->shape();
  auto rank = shapeA.rank();

  if (rank < 2) {
    throw std::invalid_argument("can only transpose matrices");
  }

  std::vector<unsigned> axes(shapeA.begin(), shapeA.end());


  std::swap(axes[rank - 1], axes[rank - 2]);

  wrapComputation("Transpose", {in(0)});
  deduceStatus();
}

Transpose::Transpose(const matcha::Tensor& a)
  : Transpose(deref(a))
{}

/*

const NodeLoader* Transpose::getLoader() const {
  return loader();
}

const NodeLoader* Transpose::loader() {
  static NodeLoader nl = {
    .type = "Transpose",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 1) throw std::invalid_argument("loading Transpose: incorrect number of arguments");
      return new Transpose(ins[0]);
    }
  };
  return &nl;
};

*/

}
}
}
