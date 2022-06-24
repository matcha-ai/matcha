#include "bits_of_matcha/engine/flow/module/Module.h"
#include "bits_of_matcha/engine/flow/module/ModuleForw.h"
#include "bits_of_matcha/engine/flow/module/ModuleBack.h"
#include "bits_of_matcha/engine/autograd/Autograd.h"
#include "bits_of_matcha/engine/autograd/AccumulateGrads.h"
#include "bits_of_matcha/engine/tensor/factories.h"

#include <ranges>


namespace matcha::engine {

Module::Module(std::unique_ptr<Graph>&& graph)
{
  graph_ = std::move(graph);
}

std::vector<Tensor*> Module::forward(const std::vector<Tensor*>& ins) {
  std::vector<Tensor*> result;
  result.reserve(graph_->outputs.size());
  for (auto gout: graph_->outputs) {
    auto out = new Tensor(gout->frame());
    gout->shareBuffer(out);
    result.push_back(gout);
  }

  forward(ins, result);
  return result;
}

void Module::forward(const std::vector<Tensor*>& ins,
                     std::vector<Tensor*>& outs)
{
  stream(ins, graph_->inputs);
  for (auto op: graph_->ops) op->run();
  stream(graph_->outputs, outs);
}

std::vector<Tensor*> Module::forward(const std::vector<Tensor*>& inputs,
                                     Partials& partials,
                                     const std::vector<Tensor*>& wrt)
{
std::vector<Tensor*> result;
  result.reserve(graph_->outputs.size());
  for (auto gout: graph_->outputs) {
    auto out = new Tensor(gout->frame());
    result.push_back(out);
  }
  forward(inputs, result, partials, wrt);
  return result;
}

void Module::forward(const std::vector<Tensor*>& inputs,
                     std::vector<Tensor*>& outputs,
                     Partials& partials,
                     const std::vector<Tensor*>& wrt)
{
  for (auto w: wrt)
    partials[w] = {nullptr, {}};

  stream(inputs, graph_->inputs);

  for (auto op: graph_->ops) {
    if (typeid(*op) == typeid(ModuleForw)) {
      forwardModuleOp(dynamic_cast<ModuleForw*>(op), partials);
    } else {
      forwardRegularOp(op, partials);
    }
  }

  stream(graph_->outputs, outputs);
}

void Module::forwardRegularOp(Op* op, Partials& partials) {
  bool backprop = false;
  for (auto in: op->inputs) {
    if (partials.contains(in)) {
      backprop = true;
      break;
    }
  }
  op->run();
  if (backprop) {
    for (auto out: op->outputs) {
      partials[out] = {nullptr, {}};
    }
  }
}

void Module::forwardModuleOp(ModuleForw* op, Partials& partials) {
  auto module = op->module_;
  auto& graph = module->graph_;
  for (int i = 0; i < graph->inputs.size(); i++) {
    auto gin = graph->inputs[i];
    auto in = op->inputs[i];
    if (partials.contains(in)) {
      partials[gin] = {nullptr, {}};
    }
  }
  module->forward(op->inputs.stdVector(), op->outputs.stdVector(), partials);
  for (int i = 0; i < graph->outputs.size(); i++) {
    auto gout = graph->outputs[i];
    auto out = op->outputs[i];
    if (partials.contains(gout)) {
      partials[out] = {nullptr, {}};
    }
  }
}

void Module::backward(Partials& partials, const std::vector<Tensor*>& deltas) {
  for (int i = 0; i < deltas.size(); i++) {
    auto d = deltas[i];
    auto out = graph_->outputs[i];
    partials[out].first = d;
  }

  for (int i = (int) graph_->ops.size() - 1; i >= 0; i--) {
    auto op = graph_->ops[i];
    if (typeid(*op) == typeid(ModuleForw)) {
      backwardModuleOp(dynamic_cast<ModuleForw*>(op), partials);
    } else {
      backwardRegularOp(op, partials);
    }
  }
}

void Module::backwardRegularOp(Op* op, Partials& partials) {
//  print("=========");
//  print(op);
  BackCtx ctx { .forward = op };
  bool required = false;
  for (auto in: op->inputs) {
    if (partials.contains(in)) {
      required = true;
      auto w = new Tensor(Float, in->shape());
      ctx.wrts.push_back(w);
    } else {
      ctx.wrts.push_back(nullptr);
    }
  }
  if (!required) return;
  for (auto out: op->outputs) {
//    print("local val: ", out);
    auto& partial = partials[out];
    accumulateGrads(partial, out->shape());
    ctx.vals.push_back(partial.first);
  }
//  for (auto v: ctx.vals) std::cout << v << " ";
//  std::cout << std::endl;
  auto back = ops::back(ctx);
//  for (auto w: ctx.wrts) std::cout << w << " ";
//  std::cout << std::endl;
//  print("--------");
  if (!back) return;
//  print(ops::name(back));
  back->run();
  for (int i = 0; i < op->inputs.size(); i++) {
    auto in = op->inputs[i];
    auto w = ctx.wrts[i];
    if (!w) continue;
    partials[in].second.push_back(w);
  }
}

void Module::backwardModuleOp(ModuleForw* op, Partials& partials) {
//  print("=============================================");
  bool required = false;
  auto module = op->module_;
  for (int i = 0; i < op->outputs.size(); i++) {
    auto out = op->outputs[i];
    auto gout = module->graph_->outputs[i];
    if (!partials.contains(gout)) continue;
    required = true;

//    print(ops::name(gout->op()));
//    print(gout->op());
    auto& partial = partials[out];
//    print("local delta: ", gout, " ", out);
    accumulateGrads(partial, out->shape());
    auto& gpartial = partials[gout];
    gpartial.second.push_back(partial.first);
  }
  if (!required) return;
//  print("Module Back!!!!!!!!!!!!!!!!!!!!!!!!");
  module->backward(partials);
  for (int i = 0; i < op->inputs.size(); i++) {
    auto in = op->inputs[i];
    auto gin = module->graph_->inputs[i];
    if (!partials.contains(gin)) continue;
    auto& gpartial = partials[gin];
    accumulateGrads(gpartial, in->shape());
    auto& partial = partials[in];
    partial.second.push_back(gpartial.first);
  }

}

void Module::accumulateGrads(Partial& partial, const Shape& shape) {
  auto& [target, grads] = partial;
  if (!target) target = new Tensor(Float, shape);
//  for (auto g: grads) std::cout << g << " ";
//  std::cout << "-> " << target << std::endl;
  if (target->buffer()) return;
  auto acc = new AccumulateGrads(grads, target);
//  print(ops::name(acc));
  acc->init();
  acc->run();
  delete acc;
}

void Module::stream(const std::vector<Tensor*>& source,
                    std::vector<Tensor*>& target)
{
  if (source.size() != target.size())
    throw std::runtime_error("source and target count mismatch");
  for (int i = 0; i < source.size(); i++)
    target[i]->shareBuffer(source[i]);
}

}