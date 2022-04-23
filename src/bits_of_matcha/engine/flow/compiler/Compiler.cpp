#include "bits_of_matcha/engine/flow/compiler/Compiler.h"
#include "bits_of_matcha/engine/autograd/AccumulateGrads.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/engine/tensor/factories.h"
#include "bits_of_matcha/engine/utils/TexGraph.h"

#include <set>
#include <map>


namespace matcha::engine {

Compiler::Compiler(Graph* graph, const std::map<tensor*, Tensor*>& grads)
  : graph_(graph)
  , grads_(grads)
  , gradsMask_(graph)
{}

Tasks compile(Graph* graph, const std::map<tensor*, Tensor*>& grads) {
  Compiler compiler(graph, grads);
  return compiler.run();
}

Tasks compile(Graph& graph, const std::map<tensor*, Tensor*>& grads) {
  return compile(&graph, grads);
}

Tasks Compiler::run() {
  auto ctx = graph_->ctx();
  for (auto [external, target]: grads_) gradsMask_[target] = true;
  auto backwardGraph = buildBackwardGraph();
  const bool tex = false;

  if (tex) {
    std::cout << utils::TexGraph {
    .graph = graph_,
    .claimGraphCtx = false,
    .tensorInfo = [&](auto tensor) {
      utils::TexGraph::TensorInfo info;
      if (backwardGraph.adjointTensors[tensor]) info.color = "orange";
      return info;
    },
    .includeFooter = false,
    };
  }

  backwardGraph.adjointGraph->claimCtx();
  TensorMask deltas(backwardGraph.adjointGraph);
  for (auto delta: backwardGraph.adjointGraph->inputs) deltas[delta] = true;

  if (tex) {
    std::cout << utils::TexGraph {
    .graph = backwardGraph.adjointGraph,
    .claimGraphCtx = false,
    .caption = "Backpropagation",
    .tensorInfo = [&, gradTargets = gradsMask_.get()](auto tensor) {
      utils::TexGraph::TensorInfo info;
      if (deltas[tensor]) info.label = "$\\Delta$";
      for (auto target: gradTargets) {
        if (backwardGraph.adjointTensors[target] == tensor) info.color = "red!40";
      }
      return info;
    },
    .includeHeader = false,
    };
  }

  Tasks tasks = generateTasks(backwardGraph);
  tasks.init();
  return tasks;
}

Tasks Compiler::generateTasks(const AdjointGraph& back) {
  std::map<Buffer*, std::vector<Tensor*>> allocations;
  std::vector<std::function<void ()>> forward, backward;
  auto effects = findEffects().get();
  const bool debug = false;

  TensorDict<unsigned> forwardTensorReqs(graph_);
  for (auto tensor: graph_->tensors) {
    auto& reqs = forwardTensorReqs[tensor];
    reqs = tensor->reqs();
    if (back.adjointTensors[tensor]) {
      reqs += 1;
    }
  }

  TensorDict<unsigned> backwardTensorReqs(back.adjointGraph);
  for (auto tensor: back.adjointGraph->tensors) {
    auto& reqs = backwardTensorReqs[tensor];
    reqs = tensor->reqs();
//    print(tensor , " has ", reqs, " reqs");
  }

  for (auto op: effects) {
//    op->init();
    if (debug)
      forward.emplace_back([=] { print("run  ", op, " (", ops::name(op), ")"); op->run(); });
    else
      forward.emplace_back([=] { op->run(); });

    for (auto i: op->inputs) {
      if (!i) continue;
      if (!i->op()) continue;

      auto& remaining = forwardTensorReqs[i];
      if (!remaining) throw std::runtime_error("lifetime is already 0");
      if (!--remaining) {
//        forward.emplace_back([=] {print("free ", i, " (forward intermediate, ", i->bytes(), " bytes)"); i->free();});
//        forward.emplace_back([=] {i->free();});
      }
    }
  }

  for (auto op: back.adjointGraph->ops) {
//    print(ops::name(op));
//    op->init();
    if (debug)
      backward.emplace_back([=] { print("run  ", op, " (", ops::name(op), ")"); op->run(); });
    else
      backward.emplace_back([=] { op->run(); });

    for (auto i: op->inputs) {
      if (!i) continue;
      auto& remaining = backwardTensorReqs[i];
      if (!remaining) throw std::runtime_error("lifetime is already 0");
      if (!--remaining) {
        if (debug)
          backward.emplace_back([=] { print("free ", i, " (backprop intermediate, ", i->bytes(), " bytes)"); i->free(); });
        else
          backward.emplace_back([=] { i->free(); });
      }
    }
  }

  for (auto tensor: graph_->tensors) {
    auto& reqs = forwardTensorReqs[tensor];
    if (!reqs) continue;
    if (debug)
      backward.emplace_back([=] { print("free ", tensor, " (forward cache, ", tensor->bytes(), " bytes)"); tensor->free(); });
    else
      backward.emplace_back([=] { tensor->free(); });
  }

  std::map<tensor*, Tensor*> backTargets = grads_;
  for (auto& [external, target]: backTargets) {
    target = back.adjointTensors[target];
  }

  Tensor* delta = back.adjointGraph->inputs.empty()
    ? nullptr
    : back.adjointGraph->inputs[0];

  Tasks tasks {
    .instructionsForward = forward,
    .instructionsBackward = backward,
    .inputs = graph_->inputs,
    .outputs = graph_->outputs,
    .delta = delta,
    .grads = backTargets,
  };
  return tasks;
}

TensorDict<unsigned> Compiler::getTotalTensorReqs(const AdjointGraph& back) {
  TensorDict<unsigned> reqsRemaining(graph_);
  for (auto tensor: graph_->tensors) {
    auto& lifetime = reqsRemaining[tensor];
    reqsRemaining[tensor] = tensor->reqs();
    if (back.adjointTensors[tensor]) {
      reqsRemaining[tensor] += back.adjointTensors[tensor]->reqs();
    }
  }

  return reqsRemaining;
}

OpMask Compiler::findEffects() {
  OpMask effects(graph_);
  for (auto out: graph_->outputs) {
    auto op = out->op();
    if (!op) continue;
    graph_->dfsPostfix(op, [](auto){}, effects);
  }

  for (auto op: graph_->ops) {
    if (ops::isSideEffect(op)) {
      effects[op] = true;
    }
  }

  return effects;
}

AdjointGraph Compiler::buildBackwardGraph() {
  TensorMask partialsMask = findGradientFlow();
  TensorMask intermediate = partialsMask & ~gradsMask_;
  AdjointGraph adjointGraph {
    .adjointGraph = new Graph(),
    .adjointTensors = TensorDict<Tensor*>(graph_),
    .adjointOps = OpDict<Op*>(graph_)
  };

  if (partialsMask.count() == 0) return adjointGraph;

  // partials to accumulate
  using Partials = std::pair<Tensor*, std::vector<Tensor*>>;
  TensorDict<Partials> partials(graph_);

  Tensor* altitude = graph_->outputs[0];
  Tensor* delta = ones({});

  partials[altitude].first = delta;
  OpDict<Op*> adjointOps(graph_);

  for (auto partial: partialsMask.get()) {
    if (partial == altitude) continue;
    auto t = new Tensor(partial->frame());
    partials[partial].first = t;
  }

  for (auto tensor: intermediate.rget()) {
    auto op = tensor->op();
    if (!op || adjointOps[op]) continue;
    std::vector<Tensor*> localPartials;
    std::vector<Tensor*> localWrts;

    for (auto out: op->outputs) {
//      print("local partial: ", partials[out].first);
      localPartials.push_back(partials[out].first);
    }

    for (auto in: op->inputs) {
      if (partialsMask[in]) {
        auto t = new Tensor(in->frame());
        localWrts.push_back(t);
        partials[in].second.push_back(t);
      } else {
        localWrts.push_back(nullptr);
      }
    }

    auto back = ops::back({
      .forward = op,
      .vals = localPartials,
      .wrts = localWrts
    });

    if (!back) continue;
    adjointOps[op] = back;
  }

  auto graph = new Graph();

  std::set<Tensor*> addedTensors;
  std::set<Op*> addedOps;

  auto tryAddTensor = [&] (Tensor* tensor) {
    if (addedTensors.contains(tensor)) return;
    addedTensors.insert(tensor);
    graph->tensors.push_back(tensor);
    tensor->ctx().setMode(TensorCtx::Constant);
  };

  graph->inputs.push_back(delta);

  for (Tensor* t: partialsMask.rget()) {
    auto& partial = partials[t];
    Op* adjoint = t->op() ? adjointOps[t->op()] : nullptr;

    std::vector<Tensor*>& grads = partial.second;
    Tensor* target = partial.first;

    if (adjoint && !addedOps.contains(adjoint)) {
//      print("Adding ", adjoint);
      graph->ops.push_back(adjoint);
      addedOps.insert(adjoint);
      adjoint->ctx().setTraced();
    }

    if (t == altitude) {
      tryAddTensor(delta);
      continue;
    }

    auto accumulator = new autograd::AccumulateGrads(grads, target);
    for (auto grad: grads) tryAddTensor(grad);
    tryAddTensor(target);

    accumulator->ctx().setTraced();
    graph->ops.push_back(accumulator);
  }

  TensorDict<Tensor*> adjointTensors(graph_);
  for (auto tensor: graph_->tensors) adjointTensors[tensor] = partials[tensor].first;

  adjointGraph = AdjointGraph {
    .adjointGraph = graph,
    .adjointTensors = adjointTensors,
    .adjointOps = adjointOps,
  };

  return adjointGraph;
}

TensorMask Compiler::findGradientFlow() {
  TensorMask partialsMask(graph_);
  Tensor* altitude = graph_->outputs[0];
  for (auto target: gradsMask_.get()) {
    partialsMask |= graph_->between(target, graph_->outputs[0]);
  }
  return partialsMask;
}

}