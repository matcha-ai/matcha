#include "bits_of_matcha/fn/transpose.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/tuple.h"
#include "bits_of_matcha/engine/tensor.h"

#include <matcha/device>


namespace matcha {
namespace fn {

Tensor transpose(const Tensor& a) {
  auto* node = new engine::fn::Transpose(a);
  return Tensor::fromObject(node->out(0));
}

}
}


namespace matcha {
namespace engine {
namespace fn {

Transpose::Transpose(Tensor* a)
  : Fn{a}
{
  auto& shapeA = in(0)->shape();
  auto rank = shapeA.rank();

  if (rank < 2) {
    throw std::invalid_argument("can only transpose matrices");
  }

  std::vector<unsigned> axes(shapeA.begin(), shapeA.end());


  std::swap(axes[rank - 1], axes[rank - 2]);

  addOut(in(0)->dtype(), Shape(axes));
  computation_ = device::Cpu().createComputation(
     "Transpose",
     {in(0)->buffer()}
  );
  computation_->prepare();
  out(0)->setBuffer(computation_->target(0));
}

Transpose::Transpose(const matcha::Tensor& a)
  : Transpose(deref(a))
{}

void Transpose::eval(Tensor* target) {
  if (!required()) return;
  unrequire();
  evalIns();
  computation_->run();
}

const NodeLoader* Transpose::getLoader() const {
  return loader();
}

const NodeLoader* Transpose::loader() {
  static NodeLoader nl = {
    .type = "Transpose",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 1) throw std::invalid_argument("loading Transpose: incorrect number of arguments");
      return new Transpose(ins[0]);
    }
  };
  return &nl;
};

}
}
}
