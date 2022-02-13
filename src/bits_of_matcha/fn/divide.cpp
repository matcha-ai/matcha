#include "bits_of_matcha/fn/divide.h"

#include <matcha/engine>


namespace matcha {
namespace fn {

Tensor divide(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::Divide(a, b);
  auto* out  = new engine::Tensor(node->out(0));
  return Tensor::fromObject(out);
}

UnaryFn divideWith(const Tensor& b) {
  return [=](auto& a) {
    return a / b;
  };
}

UnaryFn divideAgainst(const Tensor& a) {
  return [=](auto& b) {
    return a / b;
  };
}

}
}

matcha::Tensor operator/(const matcha::Tensor& a, const matcha::Tensor& b) {
  return matcha::fn::divide(a, b);
}


const matcha::Tensor& operator/=(matcha::Tensor& a, const matcha::Tensor& b) {
  a = a / b;
  return a;
}


namespace matcha {
namespace engine {
namespace fn {

Divide::Divide(Tensor* a, Tensor* b)
  : Fn{a, b}
{
  if (in(0)->rank() == 0) {
    wrapComputation("DivideScalar0", {in(0), in(1)});
  } else if (in(1)->rank() == 0) {
    wrapComputation("DivideScalar1", {in(0), in(1)});
  } else if (in(0)->shape() == in(1)->shape()) {
    wrapComputation("DivideMatching", {in(0), in(1)});
  } else {
    throw std::invalid_argument("shapeA != shapeB");
  }

  deduceStatus();
}

Divide::Divide(const matcha::Tensor& a, const matcha::Tensor& b)
  : Divide(deref(a), deref(b))
{}

/*

const NodeLoader* Divide::getLoader() const {
  return loader();
}

const NodeLoader* Divide::loader() {
  static NodeLoader nl = {
    .type = "Divide",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 2) throw std::invalid_argument("loading Divide: incorrect number of arguments");
      return new Divide(ins[0], ins[1]);
    }
  };
  return &nl;
};

*/

}
}
}
