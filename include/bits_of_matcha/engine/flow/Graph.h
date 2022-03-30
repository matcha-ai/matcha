#pragma once

#include <vector>
#include <functional>
#include <map>


namespace matcha::engine {

class Node;
class Tensor;

struct Graph {
  std::vector<Node*> nodes;
  std::vector<Tensor*> tensors;
  std::vector<Tensor*> ins;
  std::vector<Tensor*> outs;

  void initCtx();

  class NodeMask {
  public:
    explicit NodeMask(Graph* graph, bool defaultValue = false);
    ~NodeMask();

    bool* begin();
    bool* end();

    const bool* begin() const;
    const bool* end() const;

    bool& operator[](int idx);
    bool& operator[](Node* node);
    NodeMask operator~();
    NodeMask operator&(const NodeMask& mask);
    NodeMask operator|(const NodeMask& mask);
    NodeMask operator&=(const NodeMask& mask);
    NodeMask operator|=(const NodeMask& mask);

    std::vector<Node*> get();
    std::vector<Node*> rget();

    size_t size() const;

  private:
    bool* data_;
    Graph* graph_;
  };

  class TensorMask {
  public:
    explicit TensorMask(Graph* graph, bool defaultValue = false);
    ~TensorMask();

    bool* begin();
    bool* end();

    const bool* begin() const;
    const bool* end() const;

    bool& operator[](int idx);
    bool& operator[](Tensor* tensor);
    TensorMask operator~();
    TensorMask operator&(const TensorMask& mask);
    TensorMask operator|(const TensorMask& mask);
    TensorMask& operator&=(const TensorMask& mask);
    TensorMask& operator|=(const TensorMask& mask);

    std::vector<Tensor*> get();
    std::vector<Tensor*> rget();

    size_t size() const;

  private:
    bool* data_;
    Graph* graph_;
  };

  NodeMask effects();

  using NodeCallback = std::function<void (Node*)>;
  void dfsPrefix(Node* root, const NodeCallback& callback);
  void dfsPostfix(Node* root, const NodeCallback& callback);
  void dfsPrefix(Node* root, const NodeCallback& callback, NodeMask& visited);
  void dfsPostfix(Node* root, const NodeCallback& callback, NodeMask& visited);

  using TensorCallback = std::function<void (Tensor*)>;
  void dfsPrefix(Tensor* root, const TensorCallback& callback);
  void dfsPostfix(Tensor* root, const TensorCallback& callback);
  void dfsPrefix(Tensor* root, const TensorCallback& callback, TensorMask& visited);
  void dfsPostfix(Tensor* root, const TensorCallback& callback, TensorMask& visited);


  TensorMask between(Tensor* source, Tensor* target);
};


}