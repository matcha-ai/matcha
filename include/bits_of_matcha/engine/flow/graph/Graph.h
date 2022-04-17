#pragma once

#include <vector>
#include <functional>

namespace matcha::engine {

class Op;

template <class T>
class OpDict;
class OpMask;

class Tensor;

template <class T>
class TensorDict;
class TensorMask;

struct Graph {
  std::vector<Tensor*> tensors;
  std::vector<Tensor*> inputs;
  std::vector<Tensor*> outputs;
  std::vector<Op*> ops;

  class CtxGuard {
  public:
    ~CtxGuard();

  private:
    explicit CtxGuard(Graph* graph);
    Graph* graph_;

    friend class Graph;
  };
  CtxGuard ctx();

  Graph unfold();

  using OpCallback = std::function<void (Op*)>;
  void dfsPrefix(Op* root, const OpCallback& callback);
  void dfsPostfix(Op* root, const OpCallback& callback);
  void dfsPrefix(Op* root, const OpCallback& callback, OpMask& visited);
  void dfsPostfix(Op* root, const OpCallback& callback, OpMask& visited);
  void dfsPrefix(Op* root, const OpCallback& callback, OpDict<int>& visitedDepth);
  void dfsPostfix(Op* root, const OpCallback& callback, OpDict<int>& visitedDepth);

  using TensorCallback = std::function<void (Tensor*)>;
  void dfsPrefix(Tensor* root, const TensorCallback& callback);
  void dfsPostfix(Tensor* root, const TensorCallback& callback);
  void dfsPrefix(Tensor* root, const TensorCallback& callback, TensorMask& visited);
  void dfsPostfix(Tensor* root, const TensorCallback& callback, TensorMask& visited);
  void dfsPrefix(Tensor* root, const TensorCallback& callback, TensorDict<int>& visitedDepth);
  void dfsPostfix(Tensor* root, const TensorCallback& callback, TensorDict<int>& visitedDepth);

  TensorMask between(Tensor* source, Tensor* target);

  void claimCtx();
  void unclaimCtx();
};

}