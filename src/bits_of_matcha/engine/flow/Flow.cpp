#include "bits_of_matcha/engine/flow/Flow.h"
#include "bits_of_matcha/engine/Tensor.h"
#include "bits_of_matcha/engine/Node.h"
#include "bits_of_matcha/engine/NodeBacward.h"
#include "bits_of_matcha/print.h"


namespace matcha::engine {

Flow::Flow(Graph graph)
  : graph_{std::move(graph)}
{}

std::vector<Tensor*> Flow::run(const std::vector<Tensor*> ins) {
//  instructions_.run();
  return {};
}

void Flow::compile() {
  createBackwardFlow();
}

void Flow::createBackwardFlow() {
  Graph::TensorMask targetsMask(&graph_);
  targetsMask[graph_.ins[0]] = true;
  Graph::TensorMask tensorsMask(&graph_);

  // get all tensors that will be in backward flow
  for (auto wrt: targetsMask.get()) {
    auto between =  graph_.between(wrt, graph_.outs[0]);
    tensorsMask |= between;
  }

  // get only intermediate chain tensors
  auto intermediatesMask = tensorsMask & ~targetsMask;
  Graph::NodeMask nodesMask(&graph_);

  // intermediate chain tensors sources
  for (auto tensor: intermediatesMask.get()) {
    nodesMask[tensor->source()] = true;
  }

  std::vector<NodeBackward*> nodes(graph_.nodes.size(), nullptr);
  std::vector<Tensor*> tensors(graph_.nodes.size(), nullptr);

  tensors[graph_.outs[0]->ctxId()] = Tensor::full(1, {});

  // in reverse topological ordering
  for (auto node: nodesMask.rget()) {
    std::vector<Tensor*> chainIns(node->degOut());
    for (int i = 0; i < node->degOut(); i++) {
      auto t = tensors[node->out(i)->ctxId()];
      if (!t) t = Tensor::full(0, {});
      chainIns[i] = t;
    }
    print(nodesMask.get().size());

    NodeBackward::ArgPartials partials;
    for (int i = 0; i < node->degIn(); i++) {
      if (tensorsMask[node->in(i)]) partials.emplace_back(i);
    }


    auto backward = node->createBackward(chainIns, partials);
    nodes[node->ctxId()] = backward;
    if (!backward) continue;

    for (auto out: backward->outs()) {

    }

  }

}

}
