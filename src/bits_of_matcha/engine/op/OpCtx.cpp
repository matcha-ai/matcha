#include "bits_of_matcha/engine/op/OpCtx.h"
#include "bits_of_matcha/engine/flow/Tracer.h"
#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/print.h"


namespace matcha::engine {

OpCtx::OpCtx(Op* op) {
  data_ = -2;
}

bool OpCtx::untraced() {
  return data_ == -2;
}

bool OpCtx::traced() {
  return data_ != -2;
}

int OpCtx::key() const {
  return data_;
}

void OpCtx::fixKey(int key) {
  if (keyFixed()) throw std::runtime_error("OpCtx key is fixed");
  data_ = key;
}

void OpCtx::unfixKey() {
  data_ = -1;
}

bool OpCtx::keyFixed() const {
  return data_ != -1;
}

void OpCtx::setTraced() {
  if (traced()) throw std::runtime_error("already traced");
  data_ = -1;
}

}