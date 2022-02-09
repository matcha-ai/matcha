#include "bits_of_matcha/fn/maxIn.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/tuple.h"
#include "bits_of_matcha/engine/tensor.h"

#include <matcha/device>


namespace matcha {
namespace fn {

Tensor maxIn(const Tensor& a) {
  auto* node = new engine::fn::MaxIn(a);
  return Tensor::fromObject(node->out(0));
}

}
}

namespace matcha {
namespace engine {
namespace fn {

MaxIn::MaxIn(Tensor* a)
  : Fn{a}
{
  addOut(in(0)->dtype(), {});
  computation_ = device::Cpu().createComputation(
     "MaxIn",
     {in(0)->buffer()}
  );
  computation_->prepare();
  out(0)->setBuffer(computation_->target(0));
}

MaxIn::MaxIn(const matcha::Tensor& a)
  : MaxIn(deref(a))
{}

void MaxIn::eval(Tensor* target) {
  if (!required()) return;
  unrequire();
  evalIns();
  computation_->run();
}

const NodeLoader* MaxIn::getLoader() const {
  return loader();
}

const NodeLoader* MaxIn::loader() {
  static NodeLoader nl = {
    .type = "MaxIn",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 1) throw std::invalid_argument("loading MaxIn: incorrect number of arguments");
      return new MaxIn(ins[0]);
    }
  };
  return &nl;
};

}
}
}
