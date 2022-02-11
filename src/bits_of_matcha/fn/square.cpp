#include "bits_of_matcha/fn/square.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/tuple.h"
#include "bits_of_matcha/engine/tensor.h"

#include <matcha/device>


namespace matcha {
namespace fn {

Tensor square(const Tensor& a) {
  auto* node = new engine::fn::Square(a);
  auto* out  = new engine::Tensor(node->out(0));
  return Tensor::fromObject(out);
}

}
}


namespace matcha {
namespace engine {
namespace fn {

Square::Square(Tensor* a)
  : Fn{a}
{
  wrapComputation("Square", {in(0)});
  deduceStatus();
}

Square::Square(const matcha::Tensor& a)
  : Square(deref(a))
{}

/*

const NodeLoader* Square::getLoader() const {
  return loader();
}

const NodeLoader* Square::loader() {
  static NodeLoader nl = {
    .type = "Square",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 1) throw std::invalid_argument("loading Square: incorrect number of arguments");
      return new Square(ins[0]);
    }
  };
  return &nl;
};

*/

}
}
}
