#include "bits_of_matcha/fn/sum.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/tuple.h"
#include "bits_of_matcha/engine/tensor.h"

#include <matcha/device>


namespace matcha {
namespace fn {

Tensor sum(const Tensor& a) {
  auto* node = new engine::fn::Sum(a);
  return Tensor::fromObject(node->out(0));
}

}
}

namespace matcha {
namespace engine {
namespace fn {

Sum::Sum(Tensor* a)
  : Fn{a}
{
  addOut(in(0)->dtype(), {});
  computation_ = device::Cpu().createComputation(
     "Sum",
     {in(0)->buffer()}
  );
  computation_->prepare();
  out(0)->setBuffer(computation_->target(0));
}

Sum::Sum(const matcha::Tensor& a)
  : Sum(deref(a))
{}

void Sum::eval(Tensor* target) {
  if (!required()) return;
  unrequire();
  evalIns();
  computation_->run();
}

const NodeLoader* Sum::getLoader() const {
  return loader();
}

const NodeLoader* Sum::loader() {
  static NodeLoader nl = {
    .type = "Sum",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 1) throw std::invalid_argument("loading Sum: incorrect number of arguments");
      return new Sum(ins[0]);
    }
  };
  return &nl;
};

}
}
}
