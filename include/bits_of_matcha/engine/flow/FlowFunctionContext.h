#pragma once

#include "bits_of_matcha/fn.h"
#include "bits_of_matcha/Flow.h"
#include "bits_of_matcha/engine/Tensor.h"

#include <string>
#include <variant>


namespace matcha::engine {


#define flow_make(function, a)                                                    \
  static auto __flowCtx__ = matcha::engine::FlowFunctionContext(function);        \
  if (!__flowCtx__.built()) {                                                     \
    if (!__flowCtx__.building()) {                                                \
      __flowCtx__.build({&(a)});                                                  \
    }                                                                             \
  }                                                                               \
  if (!__flowCtx__.building()) {                                                  \
    return __flowCtx__.run({&(a)})[0];                                            \
  }                                                                               \


#define flow_load(file)                                                           \
  __flowCtx__.load(file);                                                         \
  return __flowCtx__.run()[0];                                                    \


class Flow;

class FlowFunctionContext {
public:
  using Function = matcha::Flow::Function;
  explicit FlowFunctionContext(const Function& fn);

  void load(const std::string& file);

  bool built();
  bool building();

  void build(const std::vector<const tensor*>& ins);
  std::vector<tensor> run(const std::vector<const tensor*>& ins);
  std::vector<tensor> run();

private:
  matcha::Flow flow_;

};


}