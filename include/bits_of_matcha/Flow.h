#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/ops.h"
#include "bits_of_matcha/savers/SaveSpec.h"
#include <map>
#include <set>

#define MATCHA_FLOW_API alignas(void*)


namespace matcha {

class MATCHA_FLOW_API Flow {
public:
  tensor operator()(const tensor& a);
  tensor operator()(const tensor& a, const tensor& b);
  tensor operator()(const tensor& a, const tensor& b, const tensor& c);
  tuple operator()(const tuple& inputs);

  void save(const std::string& file, const SaveSpec& spec = {});

  void build(const std::vector<tensor>& tensors);
  void build(const std::vector<Frame>& frames);

  size_t flops() const;
  size_t flops(const std::vector<Frame>& frames) const;
  size_t flops(const std::vector<tensor>& tensors) const;

public:
  // functional API
  Flow(const AnyOp& op);

protected:
  // subclassing API

  virtual void init(const tensor& a);
  virtual void init(const tensor& a, const tensor& b);
  virtual void init(const tensor& a, const tensor& b, const tensor& c);
  virtual void init(const tuple& inputs);

  virtual tensor run(const tensor& a);
  virtual tensor run(const tensor& a, const tensor& b);
  virtual tensor run(const tensor& a, const tensor& b, const tensor& c);
  virtual tuple run(const tuple& inputs);

public:
  Flow();
  ~Flow() = default;

private:
  void* internal_;

};

}