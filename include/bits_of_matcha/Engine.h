#pragma once


namespace matcha::engine {
class Tensor;
class Flow;
}

namespace matcha {

class tensor;
class Flow;

class Engine {
public:
  static tensor ref(engine::Tensor* a);
  static engine::Tensor* deref(const tensor& a);
  static engine::Tensor* deref(const tensor* a);
  static engine::Tensor* unref(tensor& a);
  static engine::Tensor* unref(tensor* a);
};


}