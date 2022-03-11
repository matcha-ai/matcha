#include "bits_of_matcha/engine/flowContext.h"
#include "bits_of_matcha/engine/flowBuilder.h"
#include "bits_of_matcha/engine/flow.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/print.h"

#include <future>


namespace matcha::engine {

FlowQueryResponse::FlowQueryResponse()
  : flow{nullptr}
{}

FlowQueryResponse::FlowQueryResponse(Flow* flow)
  : flow{flow}
{}

const char* FlowQueryResponse::what() const noexcept {
  return "Flow is not valid";
}

FlowContext::FlowContext(UnaryFn fn)
  : flow_{new Flow()}
  , fn_{std::move(fn)}
  , building_{false}
  , first_{false}
{
  print("FlowContext()");
}

void FlowContext::load(const std::string& path) {
  if (!flow_) throw std::runtime_error("loading into null flow");
  auto& builder = *FlowBuilder::current();
  std::cout << "loading" << std::endl;
}

bool FlowContext::built() {
  return flow_->built();
}

bool FlowContext::onBuilt() {
  return first_;
}

void FlowContext::build(const std::vector<const matcha::Tensor*>& inlet) {
  if (building_) return;
  else building_ = true;
  first_ = false;

  try {
    FlowBuilder builder(flow_);

    matcha::Tensor i0 = *inlet[0];
    matcha::Tensor o0 = fn_(i0);

    builder.inlet({i0.pimpl_});
    builder.outlet({o0.pimpl_});
    builder.finish();
    building_ = false;
    first_ = true;
  } catch (FlowQueryResponse& r) {
    building_ = false;
    first_ = true;
    if (inlet[0]->getFlowQuery()) return;

    print("HERE?");
    print(flow_);
    throw FlowQueryResponse(flow_);
  }
}

matcha::Tensor FlowContext::flow(const std::vector<const matcha::Tensor*>& inlet) {
  if (inlet[0]->getFlowQuery()) throw FlowQueryResponse(flow_);
  return 0;
}

matcha::Tensor FlowContext::respond() {
  return 0;
  std::cout << "RESPOND" << std::endl;
  throw FlowQueryResponse(flow_);
}


}