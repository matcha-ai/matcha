#include "bits_of_matcha/fn/multiply.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/tuple.h"
#include "bits_of_matcha/engine/tensor.h"

#include <matcha/device>


namespace matcha {
namespace fn {

Tensor multiply(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::Multiply(a, b);
  return Tensor::fromObject(node->out(0));
}

}
}

matcha::Tensor operator*(const matcha::Tensor& a, const matcha::Tensor& b) {
  return matcha::fn::multiply(a, b);
}


const matcha::Tensor& operator*=(matcha::Tensor& a, const matcha::Tensor& b) {
  a = a * b;
  return a;
}


namespace matcha {
namespace engine {
namespace fn {

Multiply::Multiply(Tensor* a, Tensor* b)
  : Fn{a, b}
{
  if (in(0)->rank() == 0) {

    addOut(in(1)->dtype(), in(1)->shape());
    computation_ = device::Cpu().createComputation(
       "MultiplyScalar0",
       {in(0)->buffer(), in(1)->buffer()}
    );
    computation_->prepare();
    out(0)->setBuffer(computation_->target(0));

  } else if (in(1)->rank() == 0) {

    addOut(in(0)->dtype(), in(0)->shape());
    computation_ = device::Cpu().createComputation(
       "MultiplyScalar0",
       {in(1)->buffer(), in(0)->buffer()}
    );
    computation_->prepare();
    out(0)->setBuffer(computation_->target(0));

  } else {

    if (in(0)->shape() != in(1)->shape()) {
      throw std::invalid_argument("shapeA != shapeB");
    }

    addOut(in(0)->dtype(), in(0)->shape());
    computation_ = device::Cpu().createComputation(
       "MultiplyMatching",
       {in(0)->buffer(), in(1)->buffer()}
    );
    computation_->prepare();
    out(0)->setBuffer(computation_->target(0));
  }
}

Multiply::Multiply(const matcha::Tensor& a, const matcha::Tensor& b)
  : Multiply(deref(a), deref(b))
{}

void Multiply::eval(Tensor* target) {
  if (!required()) return;
  unrequire();
  evalIns();
  computation_->run();
}

const NodeLoader* Multiply::getLoader() const {
  return loader();
}

const NodeLoader* Multiply::loader() {
  static NodeLoader nl = {
    .type = "Multiply",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 2) throw std::invalid_argument("loading Multiply: incorrect number of arguments");
      return new Multiply(ins[0], ins[1]);
    }
  };
  return &nl;
};

}
}
}
