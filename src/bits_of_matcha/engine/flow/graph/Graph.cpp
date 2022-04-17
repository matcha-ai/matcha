#include "bits_of_matcha/engine/flow/graph/Graph.h"
#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/engine/flow/graph/OpMask.h"
#include "bits_of_matcha/engine/flow/graph/OpDict.h"
#include "bits_of_matcha/engine/flow/graph/TensorMask.h"
#include "bits_of_matcha/engine/flow/graph/TensorDict.h"


namespace matcha::engine {

Graph::CtxGuard::CtxGuard(Graph* graph)
  : graph_{graph}
{
  graph_->claimCtx();
}

Graph::CtxGuard::~CtxGuard() {
  graph_->unclaimCtx();
}

Graph::CtxGuard Graph::ctx() {
  return CtxGuard(this);
}

void Graph::claimCtx() {
  int i;

  i = 0;
  for (auto op: ops) op->ctx().fixKey(i++);

  i = 0;
  for (auto tensor: tensors) tensor->ctx().fixKey(i++);
}

void Graph::unclaimCtx() {
  for (auto op: ops) op->ctx().unfixKey();
  for (auto tensor: tensors) tensor->ctx().unfixKey();
}

void Graph::dfsPostfix(Op* root, const OpCallback& callback, OpMask& visited) {
  if (!root) return;
  if (visited[root]) return;
  visited[root] = true;
  for (auto in: root->inputs) {
    auto op = in->op();
    if (op) dfsPostfix(op, callback, visited);
  }
  callback(root);
}

void Graph::dfsPostfix(Op* root, const OpCallback& callback, OpDict<int>& visited) {
  std::function<void (Op*, int)> dfs = [&](Op* root, int depth) {
    if (!root) return;
    if (visited[root] >= 0) return;
    visited[root] = depth;
    for (auto in: root->inputs) {
      auto op = in->op();
      if (op) dfs(op, depth + 1);
    }
    callback(root);
  };

  dfs(root, 0);
}

void Graph::dfsPostfix(Op* root, const OpCallback& callback) {
  OpMask visited(this);
  dfsPostfix(root, callback, visited);
}

void Graph::dfsPostfix(Tensor* root, const TensorCallback& callback) {
  TensorMask visited(this);
  dfsPostfix(root, callback, visited);
}

void Graph::dfsPostfix(Tensor* root, const TensorCallback& callback, TensorDict<int>& visitedDepth) {
  if (!root) return;
  std::function<void (Tensor*, int)> dfs = [&](Tensor* root, int depth) {
    if (visitedDepth[root] >= 0) return;
    visitedDepth[root] = depth;
    auto op = root->op();
    if (op) {
      for (auto in: op->inputs) {
        dfs(in, depth + 1);
      }
    }
    callback(root);
  };

  dfs(root, 0);
  callback(root);
}

void Graph::dfsPostfix(Tensor* root, const TensorCallback& callback, TensorMask& visited) {
  if (!root) return;
  if (visited[root]) return;
  visited[root] = true;

  auto op = root->op();
  if (op) {
    for (auto in: op->inputs) {
      dfsPostfix(in, callback, visited);
    }
  }

  callback(root);
}

TensorMask Graph::between(Tensor* source, Tensor* target) {
  TensorMask below(this);
  TensorMask above(this);

  dfsPostfix(target, [](auto) {}, below);

  above[source] = true;
  struct IsAbove : std::exception{};
  struct IsntBelow : std::exception{};

  for (auto tensor: tensors) {
    try {
      dfsPostfix(tensor, [&](auto t) {
        if (!below[t]) throw IsntBelow();
        if (above[t]) throw IsAbove();
      });
    } catch (IsAbove&) {
      above[tensor] = true;
    } catch (IsntBelow&) {
    }
  }

  return above;
}

}
