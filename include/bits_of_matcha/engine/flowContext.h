#pragma once

#include "bits_of_matcha/tensor.h"

namespace matcha::engine {

class Flow;
class FlowBuilder;

struct FlowQueryResponse : std::exception {
  FlowQueryResponse();

  FlowQueryResponse(Flow* flow);

  const char* what() const noexcept override;

  Flow* flow;
};

}

#define flow_init(function, a) \
  static auto __flowCtx__ = matcha::engine::FlowContext(function); \
  if (__flowCtx__.built()) {    \
    return __flowCtx__.flow({&(a)});                      \
  } else {                      \
    __flowCtx__.build({&(a)});                              \
  }                            \
  if (__flowCtx__.onBuilt()) {   \
    return __flowCtx__.flow({&(a)});                       \
  }

#define flow_load(file)             \
  __flowCtx__.load(file);           \
  return __flowCtx__.respond();   \


namespace matcha::engine {

class FlowContext {
  public:
    FlowContext(UnaryFn fn);
    void load(const std::string& file);

    bool built();
    bool onBuilt();

    void build(const std::vector<const matcha::Tensor*>& inlet);
    matcha::Tensor flow(const std::vector<const matcha::Tensor*>& inlet);
    matcha::Tensor respond();

  private:
    engine::Flow* flow_;
    bool building_;
    bool first_;

    UnaryFn fn_;
};

}