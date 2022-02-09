#include "bits_of_matcha/fn/exp.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/tuple.h"
#include "bits_of_matcha/engine/tensor.h"

#include <matcha/device>


namespace matcha {
namespace fn {

Tensor exp(const Tensor& a) {
  auto* node = new engine::fn::Exp(a);
  return Tensor::fromObject(node->out(0));
}

}
}


namespace matcha {
namespace engine {
namespace fn {

Exp::Exp(Tensor* a)
  : Fn{a}
{
  addOut(in(0)->dtype(), in(0)->shape());
  computation_ = device::Cpu().createComputation(
     "Exp",
     {in(0)->buffer()}
  );
  computation_->prepare();
  out(0)->setBuffer(computation_->target(0));
}

Exp::Exp(const matcha::Tensor& a)
  : Exp(deref(a))
{}

void Exp::eval(Tensor* target) {
  if (!required()) return;
  unrequire();
  evalIns();
  computation_->run();
}

const NodeLoader* Exp::getLoader() const {
  return loader();
}

const NodeLoader* Exp::loader() {
  static NodeLoader nl = {
    .type = "Exp",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 1) throw std::invalid_argument("loading Exp: incorrect number of arguments");
      return new Exp(ins[0]);
    }
  };
  return &nl;
};

}
}
}
