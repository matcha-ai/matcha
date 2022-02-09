#include "bits_of_matcha/fn/square.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/tuple.h"
#include "bits_of_matcha/engine/tensor.h"

#include <matcha/device>


namespace matcha {
namespace fn {

Tensor square(const Tensor& a) {
  auto* node = new engine::fn::Square(a);
  return Tensor::fromObject(node->out(0));
}

}
}


namespace matcha {
namespace engine {
namespace fn {

Square::Square(Tensor* a)
  : Fn{a}
{
  addOut(in(0)->dtype(), in(0)->shape());
  computation_ = device::Cpu().createComputation(
     "Square",
     {in(0)->buffer()}
  );
  computation_->prepare();
  out(0)->setBuffer(computation_->target(0));
}

Square::Square(const matcha::Tensor& a)
  : Square(deref(a))
{}

void Square::eval(Tensor* target) {
  if (!required()) return;
  unrequire();
  evalIns();
  computation_->run();
}

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

}
}
}
