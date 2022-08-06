#pragma once

#include "bits_of_matcha/engine/tensor/Tensor.h"

#include <memory>

namespace matcha::engine {

class Binding {
public:
  explicit Binding(Tensor* tensor);
  ~Binding();

  auto get() -> Tensor*;
  auto set(Tensor* tensor) -> void;

  void ref();
  void unref();
  unsigned refs() const;

private:
  Tensor* tensor_;
  unsigned refs_;
};

class View {
public:
  explicit View(Binding* binding, engine::Tensor* idx);
  explicit View(View* parent, engine::Tensor* idx);
  ~View();

  Binding* binding();
  Tensor* source();
  Tensor* idx();
  std::vector<Tensor*> indices();

  void ref();
  void unref();
  unsigned refs() const;

  const Frame& frame();
  auto read() -> engine::Tensor*;
  auto write(engine::Tensor* rhs) -> void;

private:
  engine::Tensor* idx_;
  View* parent_;
  Binding* binding_;
  engine::Tensor* cache_;
  unsigned refs_;

  engine::Tensor* cache();
};

matcha::View ref(engine::View* internal);
engine::View* deref(const matcha::View& external);
engine::View* deref(const matcha::View* external);

}