#include "bits_of_matcha/engine/ops/Assign.h"
#include "bits_of_matcha/engine/ops/Cast.h"
#include "bits_of_matcha/engine/cpu/kernels/elementwiseBinary.h"


namespace matcha::engine::ops {

Assign::Assign(Tensor* source, Tensor* target)
  : Op{source}
  , iter_(source->shape(), target->shape())
  , target_(target)
{
  if (Shape(iter_.dimsC) != target->shape())
    throw std::invalid_argument("can't broadcast shapes for assignment");

  if (target->dtype() != source->dtype()) {
    auto cast = new Cast(source, target->dtype());
    incept(this, cast);
  }
  target_->req();
}

Assign::Assign(const Assign& other)
  : Op(other)
{
  target_ = other.target_;
  target_->req();
}

Assign::~Assign() {
  target_->unreq();
}

OpMeta<Assign> Assign::meta {
  .name = "Assign",
  .sideEffect = true,
};

void Assign::run() {
  auto a = inputs[0];
  auto b = target_;
  auto c = target_;

  if (a->frame() == b->frame()) {
    b->share(a);
    return;
  }

  target_->malloc();

  switch (inputs[0]->dtype()) {
  case Float:
    cpu::elementwiseBinary<float>([](auto a, auto b) { return a; }, a->buffer(), b->buffer(), c->buffer(), iter_);
    break;
  case Double:
    cpu::elementwiseBinary<double>([](auto a, auto b) { return a; }, a->buffer(), b->buffer(), c->buffer(), iter_);
    break;
  case Int:
    cpu::elementwiseBinary<int32_t>([](auto a, auto b) { return a; }, a->buffer(), b->buffer(), c->buffer(), iter_);
    break;
  default:
    throw std::runtime_error("unsupported dtype");
  }
}

}