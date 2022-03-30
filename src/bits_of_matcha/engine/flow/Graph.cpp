#include "bits_of_matcha/engine/flow/Graph.h"
#include "bits_of_matcha/engine/Node.h"
#include "bits_of_matcha/print.h"


namespace matcha::engine {


void Graph::initCtx() {
  int i;

  i = 0;
  for (auto node: nodes) node->setCtxId(i++);
  i = 0;
  for (auto tensor: tensors) tensor->setCtxId(i++);
}

Graph::NodeMask Graph::effects() {
  Graph::NodeMask result(this);

  for (auto node: nodes) {
    result[node] = node->sideEffect();
  }

  for (auto out: outs) {
    if (!out->source()) continue;
    result[out->source()] = true;
  }

  return result;
}


void Graph::dfsPrefix(Node* root, const NodeCallback& callback, NodeMask& visited) {
  if (visited[root->ctxId()]) return;
  visited[root->ctxId()] = true;
  callback(root);
  for (int i = 0; i < root->degIn(); i++) {
    auto source = root->in(i)->source();
    if (source) dfsPrefix(source, callback, visited);
  }
}

void Graph::dfsPostfix(Node* root, const NodeCallback& callback, NodeMask& visited) {
  if (visited[root->ctxId()]) return;
  visited[root->ctxId()] = true;
  for (int i = 0; i < root->degIn(); i++) {
    auto source = root->in(i)->source();
    if (source) dfsPrefix(source, callback, visited);
  }
  callback(root);
}

void Graph::dfsPrefix(Node* root, const NodeCallback& callback) {
  NodeMask visited(this);
  dfsPrefix(root, callback, visited);
}

void Graph::dfsPostfix(Node* root, const NodeCallback& callback) {
  NodeMask visited(this);
  dfsPostfix(root, callback, visited);
}


void Graph::dfsPrefix(Tensor* root, const TensorCallback& callback, TensorMask& visited) {
  if (visited[root->ctxId()]) return;
  visited[root->ctxId()] = true;
  callback(root);
  auto node = root->source();
  if (node) {
    for (auto in: node->ins()) {
      dfsPrefix(in, callback, visited);
    }
  }
}

void Graph::dfsPostfix(Tensor* root, const TensorCallback& callback, TensorMask& visited) {
  if (visited[root->ctxId()]) return;
  visited[root->ctxId()] = true;
  auto node = root->source();
  if (node) {
    for (auto in: node->ins()) {
      dfsPostfix(in, callback, visited);
    }
  }
  callback(root);
}

void Graph::dfsPrefix(Tensor* root, const TensorCallback& callback) {
  TensorMask visited(this);
  dfsPrefix(root, callback, visited);
}

void Graph::dfsPostfix(Tensor* root, const TensorCallback& callback) {
  TensorMask visited(this);
  dfsPostfix(root, callback, visited);
}

Graph::TensorMask Graph::between(Tensor* source, Tensor* target) {
  TensorMask belowTarget(this);

  dfsPrefix(target, [](auto){}, belowTarget);
  TensorMask aboveSource(this);
  for (int i = source->ctxId(); i < nodes.size(); i++) {
    try {
      dfsPrefix(tensors[i], [&](auto tensor) {
        if (tensor == source) throw std::exception();
      }, aboveSource);
    } catch (...) {
      aboveSource[i] = true;
    }
  }

  TensorMask result = belowTarget & aboveSource;
  return result;
}


Graph::TensorMask::TensorMask(Graph* graph, bool defaultValue)
  : graph_{graph}
  , data_{nullptr}
{
  data_ = new bool[size()];
  std::fill(begin(), end(), defaultValue);
}

Graph::TensorMask::~TensorMask() {
//  print(size());
//  print(data_);
//  delete[] data_;
}

bool* Graph::TensorMask::begin() {
  return data_;
}

bool* Graph::TensorMask::end() {
  return begin() + size();
}

const bool* Graph::TensorMask::begin() const {
  return data_;
}

const bool* Graph::TensorMask::end() const {
  return begin() + size();
}

bool& Graph::TensorMask::operator[](int idx) {
  return data_[idx];
}

bool& Graph::TensorMask::operator[](Tensor* tensor) {
  return operator[](tensor->ctxId());
}

Graph::TensorMask Graph::TensorMask::operator~() {
  Graph::TensorMask result(graph_);
  std::transform(
    begin(), end(),
    result.begin(),
    [](bool t) { return !t; }
  );

  return result;
}

std::vector<Tensor*> Graph::TensorMask::get() {
  std::vector<Tensor*> result;
  for (int i = 0; i < size(); i++) {
    if (data_[i]) result.push_back(graph_->tensors[i]);
  };

  return result;
}

std::vector<Tensor*> Graph::TensorMask::rget() {
  std::vector<Tensor*> result;
  for (int i = (int) size() - 1; i >= 0; i--) {
    if (data_[i]) result.push_back(graph_->tensors[i]);
  };

  return result;
}

size_t Graph::TensorMask::size() const {
  return graph_->tensors.size();
}

Graph::TensorMask Graph::TensorMask::operator&(const TensorMask& mask) {
  Graph::TensorMask result(graph_);
  std::transform(
    begin(), end(),
    mask.begin(),
    result.begin(),
    [](bool a, bool b) { return a && b; }
  );

  return result;
}

Graph::TensorMask Graph::TensorMask::operator|(const TensorMask& mask) {
  Graph::TensorMask result(graph_);
  std::transform(
  begin(), end(),
  mask.begin(),
  result.begin(),
  [](bool a, bool b) { return a || b; }
  );

  return result;
}

Graph::TensorMask& Graph::TensorMask::operator&=(const TensorMask& mask) {
  *this = operator&(mask);
  return *this;
}

Graph::TensorMask& Graph::TensorMask::operator|=(const TensorMask& mask) {
  *this = operator|(mask);
  return *this;
}

Graph::NodeMask::NodeMask(Graph* graph, bool defaultValue)
  : graph_{graph}
{
  data_ = new bool[size()];
  std::fill(begin(), end(), defaultValue);
}

Graph::NodeMask::~NodeMask() {
  delete[] data_;
}

size_t Graph::NodeMask::size() const {
  return graph_->nodes.size();
}

bool& Graph::NodeMask::operator[](int idx) {
  return data_[idx];
}

bool& Graph::NodeMask::operator[](Node* node) {
  return operator[](node->ctxId());
}

bool* Graph::NodeMask::begin() {
  return data_;
}

bool* Graph::NodeMask::end() {
  return begin() + size();
}

const bool* Graph::NodeMask::begin() const {
  return data_;
}

const bool* Graph::NodeMask::end() const {
  return begin() + size();
}

std::vector<Node*> Graph::NodeMask::get() {
  std::vector<Node*> result;
  for (int i = 0; i < size(); i++) {
    if (data_[i]) result.push_back(graph_->nodes[i]);
  }

  return result;
}

std::vector<Node*> Graph::NodeMask::rget() {
  std::vector<Node*> result;
  for (int i = (int) size(); i >= 0; i--) {
    if (data_[i]) result.push_back(graph_->nodes[i]);
  }

  return result;
}

}