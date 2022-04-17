#include "bits_of_matcha/engine/tensor/TensorCtx.h"
#include "bits_of_matcha/engine/flow/Tracer.h"
#include "bits_of_matcha/print.h"


namespace matcha::engine {

TensorCtx::TensorCtx(Tensor* tensor) {
  mode_ = Untraced;
  key_ = -1;
}

int TensorCtx::key() const {
  return key_;
}

unsigned TensorCtx::mode() const {
  return mode_;
}

void TensorCtx::fixKey(int key) {
  if (keyFixed()) throw std::runtime_error("TensorCtx key is fixed");
  key_ = key;
}

void TensorCtx::unfixKey() {
  key_ = -1;
}

bool TensorCtx::keyFixed() const {
  return key_ != -1;
}

void TensorCtx::setMode(unsigned int mode) {
  mode_ = mode;
}

}