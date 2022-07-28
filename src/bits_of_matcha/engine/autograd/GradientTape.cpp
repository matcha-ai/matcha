#include "bits_of_matcha/engine/autograd/GradientTape.h"
#include "bits_of_matcha/engine/autograd/AccumulateGrads.h"
#include "bits_of_matcha/engine/flow/Module.h"
#include "bits_of_matcha/engine/flow/ModuleForw.h"
#include "bits_of_matcha/engine/tensor/factories.h"
#include "bits_of_matcha/engine/op/Op.h"


namespace matcha::engine {

GradientTape::GradientTape(Chain& chain, const std::vector<Tensor*>& wrt)
  : chain_(chain)
  , scopeLevel_(-1)
{
  for (auto&& w: wrt) wrt_[w] = {};
}

auto GradientTape::forward(const std::vector<Tensor*>& ins) -> std::vector<Tensor*> {
  std::vector<Tensor*> result;
  result.reserve(chain_.outputs.size());
  for (auto&& out: chain_.outputs)
    result.push_back(new Tensor(out->frame()));
  forward(ins, result);
  return result;
}

void GradientTape::forward(const std::vector<Tensor*>& ins, std::vector<Tensor*>& outs) {
  stream(ins, chain_.inputs);
  forwardInternal(chain_, {});
  stream(chain_.outputs, outs);
}

auto GradientTape::backward() -> std::vector<Tensor*> {
  std::vector<Tensor*> result;
  result.reserve(wrt_.size());
  for (auto&& [w, tmp]: wrt_)
    result.push_back(new Tensor(Float, w->shape()));
  backward(result);
  return result;
}

void GradientTape::backward(std::vector<Tensor*>& grads) {
  auto altitude = chain_.outputs[0];
  auto rootDelta = engine::ones(altitude->shape());
  backwardInternal(chain_, {rootDelta});
  print("running autograd backward");
  for (auto&& g: grads) {
    auto& partials = wrt_[g];
    auto acc = new AccumulateGrads(partials, g);
    acc->init();
    acc->run();
    delete acc;
  }
}

void GradientTape::stream(const std::vector<Tensor*>& sources, std::vector<Tensor*>& targets) {
  if (sources.size() != targets.size())
    throw std::runtime_error("can't pair sources and targets");
  for (int i = 0; i < sources.size(); i++)
    targets[i]->share(sources[i]);
}

std::vector<int> GradientTape::forwardInternal(Chain& chain, const std::vector<int>& wrtIns) {
  scopeLevel_++;
  if (scopeLevel_ >= scopes_.size()) scopes_.emplace_back();
  scopes_[scopeLevel_].emplace();
  auto& partials = getScope();
  for (auto&& [w, tmp]: wrt_) partials[w] = {nullptr, {}};
  for (auto&& i: wrtIns) partials[chain.inputs[i]] = {nullptr, {}};
  for (auto&& op: chain.ops) forwardInternal(op);
  std::vector<int> wrtOuts;
  for (int i = 0; i < chain.outputs.size(); i++)
    if (partials.contains(chain.outputs[i])) wrtOuts.push_back(i);
  scopeLevel_--;
  return wrtOuts;
}

void GradientTape::forwardInternal(Op* op) {
  auto& partials = getScope();

  if (typeid(*op) == typeid(ModuleForw)) {

    auto& chain = dynamic_cast<ModuleForw*>(op)->module().chain();
    stream(op->inputs.stdVector(), chain.inputs);
    std::vector<int> wrtIns;
    for (int i = 0; i < op->inputs.size(); i++)
      if (partials.contains(op->inputs[i])) wrtIns.push_back(i);
    auto wrtOuts = forwardInternal(chain, wrtIns);
    for (int i: wrtOuts)
      partials[op->outputs[i]] = {nullptr, {}};
    stream(chain.outputs, op->outputs.stdVector());

  } else {

    try {
//      print("running ", ops::name(op));
    } catch (...) {
//      print("running Unknown op");
    }

    op->run();
    bool required = std::any_of(op->inputs.begin(), op->inputs.end(),
                                [&](auto in) { return partials.contains(in); });

    if (!required) return;
    print("REQUIRED ", ops::name(op));
    for (auto&& out: op->outputs)
      partials[out] = {nullptr, {}};
  }
}

std::vector<Tensor*> GradientTape::backwardInternal(Chain& chain, const std::vector<Tensor*>& deltasIn) {
  scopeLevel_++;
  auto& partials = getScope();
  for (int i = 0; i < deltasIn.size(); i++)
    partials[chain.outputs[i]].first = deltasIn[i];
  for (int i = (int) chain.ops.size() - 1; i >= 0; i--)
    backwardInternal(chain.ops[i]);
  std::vector<Tensor*> deltasOut;
  for (auto&& in: chain.inputs) {
    if (!partials.contains(in)) deltasOut.push_back(nullptr);
    deltasOut.push_back(getPartial(in));
  }
  for (auto& [w, acc]: wrt_) {
    if (partials[w].second.empty()) continue;
    acc.push_back(getPartial(w));
  }
  for (auto& [t, partial]: getScope()) {
    auto& [g, acc] = partial;
    if (!wrt_.contains(t)) {
//      delete g;
      if (g && g->reqs()) g->unreq();
    }
//    for (auto&& a: acc) delete a;
  }
  scopes_[scopeLevel_].pop();
  scopeLevel_--;
  return deltasOut;
}

void GradientTape::backwardInternal(Op* op) {
  auto& partials = getScope();

  if (typeid(*op) == typeid(ModuleForw)) {

    auto& chain = dynamic_cast<ModuleForw*>(op)->module().chain();
    std::vector<Tensor*> deltasOut;
    for (auto&& out: op->outputs) {
      deltasOut.push_back(getPartial(out));
    }
    auto deltasWrt = backwardInternal(chain, deltasOut);
    for (int i = 0; i < op->inputs.size(); i++) {
      auto w = deltasWrt[i];
      if (!w) continue;
      partials[op->inputs[i]].second.push_back(w);
    }

  } else {

//    print("backward op normal");
//    print(wrt_.size());
    BackCtx ctx { .forward = op };
    bool required = false;
    for (auto&& in: op->inputs) {
      if (!partials.contains(in)) {
        ctx.wrts.push_back(false);
        continue;
      }
      ctx.wrts.push_back(true);
      required = true;
    }
    if (!required) return;
    for (auto&& out: op->outputs) {
      ctx.vals.push_back(getPartial(out));
    }
    auto bops = ops::back(ctx);
//    BackOps bops;
    if (bops.ops.empty()) {
      try {
        print("NO BOP: ", ops::name(op));
      } catch (...) {
        print("NO BOP: ???");
      }
    }
    for (auto&& bop: bops.ops) {
      try {
//      print(ops::name(op));
//        print("running ", ops::name(bop));
      } catch (...) {
//        print("running Unknown OpBack");
      }
      for (auto&& out: bop->outputs) if (out) out->req();
      bop->init();
      bop->run();
    }

    for (int i = 0; i < bops.outputs.size(); i++) {
      auto w = bops.outputs[i];
      if (!w) continue;
      w->req();
      partials[op->inputs[i]].second.push_back(w);
    }
    std::vector<Tensor*> toDelete;
    for (auto&& bop: bops.ops) {
      auto outs = bop->outputs.stdVector();
      delete bop;
      for (auto out: outs) if (out) out->unreq();
    }
  }
}

GradientTape::Partials& GradientTape::getScope() {
  return scopes_[scopeLevel_].top();
}

Tensor* GradientTape::getPartial(Tensor* t) {
  auto& partials = getScope();
  auto& [target, sources] = partials[t];
  if (!target)
    target = new Tensor(Float, t->shape());
  if (target->buffer()) return target;
  auto acc = new AccumulateGrads(sources, target);
  acc->init();
  acc->run();
  return target;
}

}