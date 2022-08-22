#include "bits_of_matcha/engine/ops/Gather.h"
#include "bits_of_matcha/engine/ops/Cast.h"
#include "bits_of_matcha/engine/cpu/kernels/fill.h"
#include "bits_of_matcha/engine/cpu/kernels/axiswiseFoldBack.h"


namespace matcha::engine::ops {

void check(Tensor* idxs) {

  switch (idxs->dtype()) {
  case Sbyte:
  case Short:
  case Int:
  case Long:
  case Byte:
  case Ushort:
  case Uint:
  case Ulong:
    break;
  default:
    throw std::invalid_argument("tensor used for indexing must be int-like");
  }
}

Gather::Gather(Tensor* a, Tensor* idxs, bool keep_dims)
  : AxiswiseFoldOp(a, keep_dims)
{
  check(idxs);
  if (idxs->size() != 1)
    throw std::runtime_error("global gather index must contain 1 element");

  if (idxs->dtype() != Int) {
    auto cast = new Cast(idxs, Int);
    incept(this, cast);
  }
  inputs.push_back(idxs);
  idxs->req();
}

Gather::Gather(Tensor* a, Tensor* idxs, int axis, bool keep_dims)
  : AxiswiseFoldOp(a, axis, keep_dims)
{
  check(idxs);
  if (idxs->size() != a->size() / a->shape()[axis])
    throw std::runtime_error("gather index size must match non-fold axes size");

  if (idxs->dtype() != Int) {
    auto cast = new Cast(idxs, Int);
    incept(this, cast);
  }
  inputs.push_back(idxs);
  idxs->req();
}

Reflection<Gather> Gather::reflection {
  .name = "Gather",
  .back = [](auto& ctx) { return dispatch<GatherBack>(ctx); },
};

void Gather::run() {
  int* idxs = inputs[1]->buffer().as<int*>();
  runCpu([=](auto* begin, auto stride, auto* end) mutable {
    if (stride == 1)
      return begin[*idxs++];
    else
      return *(begin + stride * (*idxs++));
  });
}

GatherBack::GatherBack(const BackCtx& ctx)
  : OpBack(ctx)
{
  auto fold = dynamic_cast<AxiswiseFoldOp*>(forward_);

  auto idxs = fold->inputs[1];
  inputs.push_back(idxs);
  idxs->req();
  iter_ = fold->iter();
}

Reflection<GatherBack> GatherBack::reflection {
  .name = "GatherBack",
};

void GatherBack::run() {
  auto ga = outputs[0];
  auto bga = ga->malloc();
  auto i = inputs[1];
  auto bi = i->buffer();
  auto gb = inputs[0];
  auto bgb = gb->buffer();
  auto idxs = bi.as<int*>();

  switch (ga->dtype()) {
  case Float:
    cpu::fill<float>(bga, ga->size(), 0);
    cpu::axiswiseFoldBack<float>(
      [idxs](auto* begin, auto stride, auto* end, auto gradient) mutable {
        if (stride == 1)
          begin[*idxs++] = gradient;
        else
          *(begin + stride * *(idxs++)) = gradient;
        },
      bga, bgb, iter_
    );

    break;
  default:
    throw std::runtime_error("unsupported dtype");
  }
}

}