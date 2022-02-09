#include "bits_of_matcha/fn/add.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/tuple.h"
#include "bits_of_matcha/engine/tensor.h"

#include <matcha/device>


namespace matcha {
namespace fn {

Tensor add(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::Add(a, b);
  return Tensor::fromObject(node->out(0));
}

}
}

matcha::Tensor operator+(const matcha::Tensor& a, const matcha::Tensor& b) {
  return matcha::fn::add(a, b);
}


const matcha::Tensor& operator+=(matcha::Tensor& a, const matcha::Tensor& b) {
  a = a + b;
  return a;
}


namespace matcha {
namespace engine {
namespace fn {

Add::Add(Tensor* a, Tensor* b)
  : Fn{a, b}
{
  if (in(0)->rank() == 0) {

    addOut(in(1)->dtype(), in(1)->shape());
    computation_ = device::Cpu().createComputation(
       "AddScalar0",
       {in(0)->buffer(), in(1)->buffer()}
    );
    computation_->prepare();
    out(0)->setBuffer(computation_->target(0));

  } else if (in(1)->rank() == 0) {

    addOut(in(0)->dtype(), in(0)->shape());
    computation_ = device::Cpu().createComputation(
       "AddScalar0",
       {in(1)->buffer(), in(0)->buffer()}
    );
    computation_->prepare();
    out(0)->setBuffer(computation_->target(0));

  } else if (in(0)->shape() == in(1)->shape()) {

    addOut(in(0)->dtype(), in(0)->shape());
    computation_ = device::Cpu().createComputation(
       "AddMatching",
       {in(0)->buffer(), in(1)->buffer()}
    );
    computation_->prepare();
    out(0)->setBuffer(computation_->target(0));

  } else {
    throw std::invalid_argument("shapeA != shapeB");
  }
}

Add::Add(const matcha::Tensor& a, const matcha::Tensor& b)
  : Add(deref(a), deref(b))
{}

void Add::eval(Tensor* target) {
  if (!required()) return;
  unrequire();
  evalIns();
  computation_->run();
}

const NodeLoader* Add::getLoader() const {
  return loader();
}

const NodeLoader* Add::loader() {
  static NodeLoader nl = {
    .type = "Add",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 2) throw std::invalid_argument("loading Add: incorrect number of arguments");
      return new Add(ins[0], ins[1]);
    }
  };
  return &nl;
};

}
}
}
