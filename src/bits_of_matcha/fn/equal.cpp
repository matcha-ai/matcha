#include "bits_of_matcha/fn/equal.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/tuple.h"
#include "bits_of_matcha/engine/tensor.h"

#include <matcha/device>


namespace matcha {
namespace fn {

Tensor equal(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::Equal(a, b);
  return Tensor::fromObject(node->out(0));
}

}
}

matcha::Tensor operator==(const matcha::Tensor& a, const matcha::Tensor& b) {
  return matcha::fn::equal(a, b);
}

namespace matcha {
namespace engine {
namespace fn {

Equal::Equal(Tensor* a, Tensor* b)
  : Fn{a, b}
{
  if (in(0)->rank() == 0) {

    addOut(in(1)->dtype(), in(1)->shape());
    computation_ = device::Cpu().createComputation(
       "EqualScalar0",
       {in(0)->buffer(), in(1)->buffer()}
    );
    computation_->prepare();
    out(0)->setBuffer(computation_->target(0));

  } else if (in(1)->rank() == 0) {

    addOut(in(0)->dtype(), in(0)->shape());
    computation_ = device::Cpu().createComputation(
       "EqualScalar0",
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
       "EqualMatching",
       {in(0)->buffer(), in(1)->buffer()}
    );
    computation_->prepare();
    out(0)->setBuffer(computation_->target(0));
  }
}

Equal::Equal(const matcha::Tensor& a, const matcha::Tensor& b)
  : Equal(deref(a), deref(b))
{}

void Equal::eval(Tensor* target) {
  if (!required()) return;
  unrequire();
  evalIns();
  computation_->run();
}

const NodeLoader* Equal::getLoader() const {
  return loader();
}

const NodeLoader* Equal::loader() {
  static NodeLoader nl = {
    .type = "Equal",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 2) throw std::invalid_argument("loading Equal: incorrect number of arguments");
      return new Equal(ins[0], ins[1]);
    }
  };
  return &nl;
};

}
}
}
