#include "bits_of_matcha/initializable_fn.h"
#include "bits_of_matcha/tensor.h"


namespace matcha {

InitializableUnaryFn::InitializableUnaryFn()
  : initialized_{false}
{}

tensor InitializableUnaryFn::operator()(const tensor& a) {
  if (!initialized_) {
    init(a);
    initialized_ = true;
  }
  return run(a);
}

void InitializableUnaryFn::init(const tensor& a) {
}




InitializableBinaryFn::InitializableBinaryFn()
  : initialized_{false}
{}

tensor InitializableBinaryFn::operator()(const tensor& a, const tensor& b) {
  if (!initialized_) {
    init(a, b);
    initialized_ = true;
  }
  return run(a, b);
}

void InitializableBinaryFn::init(const tensor& a, const tensor& b) {
}




InitializableTernaryFn::InitializableTernaryFn()
  : initialized_{false}
{}

tensor InitializableTernaryFn::operator()(const tensor& a, const tensor& b, const tensor& c) {
  if (!initialized_) {
    init(a, b, c);
    initialized_ = true;
  }
  return run(a, b, c);
}

void InitializableTernaryFn::init(const tensor& a, const tensor& b, const tensor& c) {
}




InitializableNaryFn::InitializableNaryFn()
  : ins_{0}
  , outs_{0}
{}

Tuple InitializableNaryFn::operator()(const Tuple& tuple) {
  if (!outs_) {
    init(tuple);
    Tuple outs = run(tuple);
    ins_ = tuple.size();
    outs_ = outs.size();
    return outs;
  }

  if (tuple.size() != ins_) throw std::invalid_argument("wrong number of inputs");
  Tuple outs = run(tuple);
  if (outs.size() != outs_) throw std::runtime_error("signature corruption");
  return outs;
}

void InitializableNaryFn::init(const Tuple& tuple) {
}


}