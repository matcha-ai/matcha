#include "bits_of_matcha/fn/matmul.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/tensor.h"
#include "bits_of_matcha/device/cpu/cpu.h"
#include "bits_of_matcha/engine/nodeloader.h"

#include <matcha/device>
#include <iostream>


namespace matcha {
namespace fn {

Tensor matmul(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::Matmul(a, b);
  return Tensor::fromObject(node->out(0));
}

}

namespace engine {
namespace fn {

Matmul::Matmul(Tensor* a, Tensor* b)
  : Fn{a, b}
{
  auto& shapeA = in(0)->shape();
  auto& shapeB = in(1)->shape();

  if (shapeA.rank() != 2 || shapeB.rank() != 2) {
    throw std::invalid_argument("Matmul: both inputs must be matrices");
  }

  unsigned rowsA = shapeA[-2];
  unsigned colsA = shapeA[-1];

  unsigned rowsB = shapeB[-2];
  unsigned colsB = shapeB[-1];

  if (colsA != rowsB) {
    throw std::invalid_argument("Matmul: colsA != rowsB");
  }

  unsigned rowsC = rowsA;
  unsigned colsC = colsB;

  addOut(Dtype::Float, {rowsC, colsC});
  computation_ = device::Cpu().createComputation(
     "Matmul",
     {in(0)->buffer(), in(1)->buffer()}
  );
  computation_->prepare();
  out(0)->setBuffer(computation_->target(0));
}

Matmul::Matmul(const matcha::Tensor& a, const matcha::Tensor& b)
  : Matmul(deref(a), deref(b))
{}

void Matmul::eval(Tensor* target) {
  if (!required()) return;
  unrequire();
  evalIns();
  computation_->run();
}

const NodeLoader* Matmul::loader() {
  static NodeLoader nl = {
    .type = "Matmul",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 2) {
        throw std::invalid_argument("loading Matmul: incorrect number of argumetns");
      }
      return new Matmul(ins[0], ins[1]);
    }
  };
  return &nl;
}

const NodeLoader* Matmul::getLoader() const {
  return loader();
}

}
}
}
