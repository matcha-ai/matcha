#include "bits_of_matcha/engine/ops/Require.h"


namespace matcha::engine::ops {

OpMeta<Require> Require::meta {
  .name = "Require",
  .sideEffect = true,
};

Require::Require(Tensor* a)
  : Op{a}
{}

Require::Require(const std::vector<Tensor*>& tensors)
  : Op(tensors)
{}

Require::Require(std::initializer_list<Tensor*> tensors)
  : Op(tensors)
{}

}