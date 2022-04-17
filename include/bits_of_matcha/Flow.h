#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/ops.h"

#define MATCHA_FLOW_API alignas(void*)


namespace matcha {

class MATCHA_FLOW_API Flow {
public:
  tensor operator()(const tensor& a);
  tensor operator()(const tensor& a, const tensor& b);
  tensor operator()(const tensor& a, const tensor& b, const tensor& c);
  tuple operator()(const tuple& inputs);

  void setRequiredGradients(const std::vector<tensor*>& tensors);
  std::vector<tensor*> requiredGradients() const;
  std::vector<std::pair<tensor*, tensor>> backward();
  int profiler();

  void save(const std::string& file);
  static Flow load(const std::string& file);

public:
  // functional API
  Flow(const AnyOp& op);

protected:
  // subclassing API
  Flow();

  virtual void init(const tensor& a);
  virtual void init(const tensor& a, const tensor& b);
  virtual void init(const tensor& a, const tensor& b, const tensor& c);
  virtual void init(const tuple& inputs);

  virtual tensor run(const tensor& a);
  virtual tensor run(const tensor& a, const tensor& b);
  virtual tensor run(const tensor& a, const tensor& b, const tensor& c);
  virtual tuple run(const tuple& inputs);

public:
  ~Flow() = default;

private:
  void* internal_;

};

Flow load(const std::string& file);

}