#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/fn.h"
#include "Device.h"

#include <variant>
#include <any>


namespace matcha {

class Flow;
class Grad;

Flow load(const std::string& filename);

class Flow {
public:
  /*
   *  Functional API
   *  Flow foo {
   *    fn1,
   *    fn2,
   *    ...
   *  };
   */
  using Function = std::variant<UnaryFn, BinaryFn, TernaryFn, NaryFn>;
  Flow(Function function);

  /*
   *  Sequential API
   *  Flow foo = [] (const tensor& a, ...) {
   *    return ...;
   *  };
   */
  Flow(std::initializer_list<UnaryFn> sequence);

protected:
  /*
   *  Subclassing API
   *  override exactly one `tensor run(...)`
   *  and optionally the appropriate `void init(...)`
   */
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

private:
  enum class API {
    Sequential,
    Functional,
    Subclassing
  };

  engine::Flow* internal_;
  Function function_;
  API api_;

private:
  struct InactiveInit : public std::exception {};
  struct InactiveRun : public std::exception {};

  template <class Fn>
  inline Fn getFunction() {
    if (!std::holds_alternative<Fn>(function_)) {
      throw std::invalid_argument("incorrect number of Flow inputs");
    }
    return std::get<Fn>(function_);
  }

};


}
