#include "bits_of_matcha/engine/flow/compiler/Compiler.h"
#include "bits_of_matcha/engine/autograd/AccumulateGrads.h"
#include "bits_of_matcha/engine/utils/TexGraph.h"

#include <set>


namespace matcha::engine {

Compiler::Compiler(Graph* graph, const TensorMask& grads)
  : graph_(graph)
  , grads_(grads)
{}

Tasks compile(Graph* graph, const TensorMask& grads) {
  Compiler compiler(graph, grads);
  return compiler.run();
}

Tasks compile(Graph& graph, const TensorMask& grads) {
  return compile(&graph, grads);
}

Tasks Compiler::run() {
  auto ctx = graph_->ctx();

  auto partials = findGradientFlow();
  Graph* backwardGraph = buildBackwardGraph();

//  print("first graph done ", backwardGraph->tensors.size(), " ", backwardGraph->ops.size(), " ", backwardGraph->inputs.size(), " ", backwardGraph->outputs.size());

  std::cout << utils::TexGraph {
    .graph = graph_,
    .claimGraphCtx = false,
    .tensorInfo = [&](auto tensor) {
      utils::TexGraph::TensorInfo info;
      if (partials[tensor]) info.color = "orange";
      return info;
    },
    .includeFooter = false,
  };

  backwardGraph->claimCtx();
  TensorMask deltas(backwardGraph);
  for (auto delta: backwardGraph->inputs) deltas[delta] = true;

  std::cout << utils::TexGraph {
    .graph = backwardGraph,
    .claimGraphCtx = false,
    .tensorInfo = [&] (auto tensor) {
      utils::TexGraph::TensorInfo info;
      if (deltas[tensor]) info.label = "$\\Delta$";
      return info;
    },
    .includeHeader = false,
  };

  Tasks tasks {
    .opsForward = graph_->ops,
    .opsBackward = backwardGraph->ops,
    .inputs  = graph_->inputs,
    .outputs = graph_->outputs,
  };

  tasks.init();
  return tasks;
}

Graph* Compiler::buildBackwardGraph() {
  TensorMask partialsMask = findGradientFlow();
  TensorMask intermediate = partialsMask & ~grads_;

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
  return graph;
}

TensorMask Compiler::findGradientFlow() {
  TensorMask partialsMask(graph_);
  Tensor* altitude = graph_->outputs[0];
  for (auto target: grads_.get()) {
    partialsMask |= graph_->between(target, graph_->outputs[0]);
  }
  return partialsMask;
}

}