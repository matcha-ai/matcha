#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/tuple.h"
#include "bits_of_matcha/fn.h"
#include "bits_of_matcha/Grad.h"
#include "Device.h"

#include <variant>
#include <any>


namespace matcha::engine {
class FlowFunctionContext;
}

namespace matcha {

class Flow;
class Grad;

Flow load(const std::string& filename);

class Flow {
public:
  using Function = std::variant<UnaryFn, BinaryFn, TernaryFn, NaryFn>;
  Flow(Function function);

protected:
  Flow();

  virtual void init(const tensor& a);
  virtual void init(const tensor& a, const tensor& b);
  virtual void init(const tensor& a, const tensor& b, const tensor& c);
  virtual void init(const Tuple& tuple);

  virtual tensor run(const tensor& a);
  virtual tensor run(const tensor& a, const tensor& b);
  virtual tensor run(const tensor& a, const tensor& b, const tensor& c);
  virtual Tuple run(const Tuple& tuple);

public:
  tensor operator()(const tensor& a);
  tensor operator()(const tensor& a, const tensor& b);
  tensor operator()(const tensor& a, const tensor& b, const tensor& c);
  Tuple operator()(const Tuple& tuple);

public:
  bool built();
  void build(const std::vector<tensor> ins);
  void build(const std::vector<Frame> ins);

public:
  Grad grad;

private:
  enum class API {
    Functional,
    Subclassing
  };

  engine::Flow* internal_;
  Function function_;
  API api_;

private:
  struct NotImplemented : std::exception {};

  template <class Fn>
  inline Fn getFunction() {
    if (!std::holds_alternative<Fn>(function_)) {
      throw std::invalid_argument("incorrect number of Flow inputs");
    }
    return std::get<Fn>(function_);
  }

  Tuple tracingFunctional(const Tuple& ins);
  Tuple tracingSubclassing(const Tuple& ins);

  friend class engine::FlowFunctionContext;

};

Flow flow(const Flow::Function& function);

}
