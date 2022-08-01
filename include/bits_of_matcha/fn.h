#pragma once

#include "bits_of_matcha/ops.h"

#include <type_traits>
#include <functional>


namespace matcha {

using unary_fn = std::function<tensor (const tensor&)>;
using binary_fn = std::function<tensor (const tensor&, const tensor&)>;
using ternary_fn = std::function<tensor (const tensor&, const tensor&, const tensor&)>;
using nary_fn = std::function<tuple (const tuple&)>;

class fn {
public:
  template <class Callable, std::enable_if_t<
  std::is_convertible<Callable, UnaryOp>() ||
  std::is_convertible<Callable, BinaryOp>() ||
  std::is_convertible<Callable, TernaryOp>() ||
  std::is_convertible<Callable, NaryOp>() ||
  std::is_same<Callable, AnyOp>()
  , bool> = true>
  fn(Callable&& callable) : internal_(callable) {}

//  fn(AnyOp function) : internal_(function) {}
//  fn(unary_fn function) : internal_(std::move(function)) {}
//  fn(BinaryOp function) : internal_(std::move(function)) {}
//  fn(TernaryOp function) : internal_(std::move(function)) {}
//  fn(NaryOp function) : internal_(std::move(function)) {}
//  explicit fn(AnyOp function) : internal_(std::move(function)) {}

  fn() = default;

  auto stdVariant() -> AnyOp& { return internal_; }
  auto stdVariant() const -> const AnyOp& { return internal_; }

  operator bool() const {
    return std::visit([](auto&& f) { return (bool) f; }, internal_);
  }

  tensor operator()(const tensor& a) const {
    return std::get<UnaryOp>(internal_)(a);
  }

  tensor operator()(const tensor& a, const tensor& b) const {
    return std::get<BinaryOp>(internal_)(a, b);
  }

  tensor operator()(const tensor& a, const tensor& b, const tensor& c) const {
    return std::get<TernaryOp>(internal_)(a, b, c);
  }

  tuple operator()(const tuple& inputs) const {
    if (std::holds_alternative<NaryOp>(internal_)) {
      return std::get<NaryOp>(internal_)(inputs);
    }

    switch (inputs.size()) {
    case 1:
      return {std::get<UnaryOp>(internal_)(inputs[0])};
    case 2:
      return {std::get<BinaryOp>(internal_)(inputs[0], inputs[2])};
    case 3:
      return {std::get<TernaryOp>(internal_)(inputs[0], inputs[2], inputs[3])};
    }

    throw std::bad_variant_access();
  }

private:
  AnyOp internal_;
};

}