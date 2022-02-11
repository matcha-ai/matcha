#include "bits_of_matcha/fn/add.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/tuple.h"
#include "bits_of_matcha/engine/tensor.h"

#include <matcha/device>


namespace matcha {
namespace fn {

Tensor add(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::Add(a, b);
  auto* out  = new engine::Tensor(node->out(0));
  return Tensor::fromObject(out);
}

}
}

matcha::Tensor operator+(const matcha::Tensor& a, const matcha::Tensor& b) {
  return matcha::fn::add(a, b);
}


const matcha::Tensor& operator+=(matcha::Tensor& a, const matcha::Tensor& b) {
  a = a + b;
  return a;
}


namespace matcha {
namespace engine {
namespace fn {


Add::Add(Tensor* a, Tensor* b)
  : Fn{a, b}
{
  if (in(0)->rank() == 0) {
    wrapComputation("AddScalar0", {in(0), in(1)});
  } else if (in(1)->rank() == 0) {
    wrapComputation("AddScalar0", {in(1), in(0)});
  } else if (in(0)->shape() == in(1)->shape()) {
    wrapComputation("AddMatching", {in(0), in(1)});
  } else {
    throw std::invalid_argument("shapeA != shapeB");
  }

  deduceStatus();
}

Add::Add(const matcha::Tensor& a, const matcha::Tensor& b)
  : Add(deref(a), deref(b))
{}

/*
const NodeLoader* Add::getLoader() const {
  return loader();
}

const NodeLoader* Add::loader() {
  static NodeLoader nl = {
    .type = "Add",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 2) throw std::invalid_argument("loading Add: incorrect number of arguments");
      return new Add(ins[0], ins[1]);
    }
  };
  return &nl;
};
*/

}
}
}
