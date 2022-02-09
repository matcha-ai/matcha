#include "bits_of_matcha/fn/subtract.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/tuple.h"
#include "bits_of_matcha/engine/tensor.h"

#include <matcha/device>


namespace matcha {
namespace fn {

Tensor subtract(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::Subtract(a, b);
  return Tensor::fromObject(node->out(0));
}

}
}

matcha::Tensor operator-(const matcha::Tensor& a, const matcha::Tensor& b) {
  return matcha::fn::subtract(a, b);
}


const matcha::Tensor& operator-=(matcha::Tensor& a, const matcha::Tensor& b) {
  a = a - b;
  return a;
}


namespace matcha {
namespace engine {
namespace fn {

Subtract::Subtract(Tensor* a, Tensor* b)
  : Fn{a, b}
{
  std::string computationName;
  if (in(0)->rank() == 0) {
    computationName = "SubtractScalar0";
    addOut(in(1)->dtype(), in(1)->shape());
  } else if (in(1)->rank() == 0) {
    computationName = "SubtractScalar1";
    addOut(in(0)->dtype(), in(0)->shape());
  } else if (in(0)->shape() == in(1)->shape()) {
    computationName = "SubtractMatching";
    addOut(in(0)->dtype(), in(0)->shape());
  } else {
    throw std::invalid_argument("shapeA != shapeB");
  }

  computation_ = device::Cpu().createComputation(
     computationName,
     {in(0)->buffer(), in(1)->buffer()}
  );
  computation_->prepare();
  out(0)->setBuffer(computation_->target(0));
}

Subtract::Subtract(const matcha::Tensor& a, const matcha::Tensor& b)
  : Subtract(deref(a), deref(b))
{}

void Subtract::eval(Tensor* target) {
  if (!required()) return;
  unrequire();
  evalIns();
  computation_->run();
}

const NodeLoader* Subtract::getLoader() const {
  return loader();
}

const NodeLoader* Subtract::loader() {
  static NodeLoader nl = {
    .type = "Subtract",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 2) throw std::invalid_argument("loading Subtract: incorrect number of arguments");
      return new Subtract(ins[0], ins[1]);
    }
  };
  return &nl;
};

}
}
}
