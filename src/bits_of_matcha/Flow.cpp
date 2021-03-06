#include "bits_of_matcha/Flow.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/engine/flow/Flow.h"
#include "bits_of_matcha/print.h"

using matcha::engine::ref;
using matcha::engine::deref;
using Internal = matcha::engine::Flow*;

namespace matcha {

Flow::Flow(const AnyOp& op)
  : internal_(new engine::Flow(op))
{

}

Flow::Flow()
  : internal_(new engine::Flow())
{}

tensor Flow::operator()(const tensor& a) {
  auto flow = Internal(internal_);

  if (!flow->hasPreimage())
    flow->setPreimage([&](const tensor& a) { init(a); return run(a); });

  auto outs = flow->call({deref(a)});

  if (outs.size() != 1)
    throw std::runtime_error("incorrect output signature");

  return ref(outs[0]);
}

void Flow::save(const std::string& file, const SaveSpec& spec) {
  auto flow = (Internal) internal_;
  FlowPlot plotspec = std::holds_alternative<FlowPlot>(spec)
  ? std::get<FlowPlot>(spec)
  : FlowPlot{};
}

void Flow::build(const std::vector<tensor>& tensors) {
  std::vector<Frame> frames;
  frames.reserve(tensors.size());
  for (auto&& t: tensors)
    frames.push_back(t.frame());

  build(frames);
}

void Flow::build(const std::vector<Frame>& frames) {
  auto flow = (Internal) internal_;
  if (!flow->hasPreimage()) {
    if (frames.size() == 1) {
      flow->setPreimage([&](const tensor& a) { init(a); return run(a); });
    }
  }

  flow->build(frames);
}

void Flow::init(const tensor& a) {}
void Flow::init(const tensor& a, const tensor& b) {}
void Flow::init(const tensor& a, const tensor& b, const tensor& c) {}
void Flow::init(const tuple& inputs) {}

struct NotSubclassed : std::exception {};
tensor Flow::run(const tensor& a) { throw NotSubclassed(); }
tensor Flow::run(const tensor& a, const tensor& b) { throw NotSubclassed(); }
tensor Flow::run(const tensor& a, const tensor& b, const tensor& c) { throw NotSubclassed(); }
tuple Flow::run(const tuple& inputs) { throw NotSubclassed(); }

}