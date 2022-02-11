#include "bits_of_matcha/engine/flow.h"
#include "bits_of_matcha/engine/flowsaver.h"
#include "bits_of_matcha/engine/flowloader.h"
#include "bits_of_matcha/engine/tensor.h"
#include "bits_of_matcha/engine/node.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/tuple.h"
#include "bits_of_matcha/flow.h"

#include <stdexcept>
#include <iostream>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>


namespace matcha {
namespace engine {

Flow::Flow(const matcha::Tuple& outs)
  : outs_{outs}
  , ins_{}
{
  deduceIns();
}

const matcha::Tensor& Flow::in(int index) const {
  if (index < 0) index += ins();
  if (index < 0 || index >= ins()) throw std::out_of_range("Flow: in index is out of range");
  return ins_[index];
}

size_t Flow::ins() const {
  return ins_.size();
}

const matcha::Tensor& Flow::out(int index) const {
  if (index < 0) index += outs();
  if (index < 0 || index >= outs()) throw std::out_of_range("Flow: out index is out of range");
  return outs_[index];
}

size_t Flow::outs() const {
  return outs_.size();
}

void Flow::save(std::ostream& os) const {
  FlowSaver::save(os, this);
}

matcha::Flow Flow::load(std::istream& is) {
  return matcha::Flow::fromObject(FlowLoader::load(is));
}

void Flow::deduceIns() const {
  std::vector<Tensor*> buffer;
  std::set<Tensor*> tensors;
  std::set<Node*> nodes;

  for (auto& out: outs_) {
    deduceIns(out.object(), buffer, tensors, nodes);
  }

  std::vector<matcha::Tensor> ins;
  ins.reserve(buffer.size());
  std::transform(
    std::begin(buffer), std::end(buffer),
    std::back_inserter(ins),
    [](Tensor* tensor) {
      return matcha::Tensor::fromObject(tensor);
    }
  );

  ins_ = Tuple(ins);
}

void Flow::deduceIns(Tensor* tensor, std::vector<Tensor*>& buffer, std::set<Tensor*>& visitedTensors, std::set<Node*>& visitedNodes) const {
  /*
  if (visitedTensors.contains(tensor)) return;
  visitedTensors.insert(tensor);

//  std::cout << "deducing In: " << tensor << std::endl;
  if (tensor->in_ == nullptr) {
//    std::cout << "no in-: " << tensor << std::endl;
    buffer.push_back(tensor);
  } else {
    // originates in a node
//    std::cout << "found Node: " << tensor << std::endl;
    deduceIns(tensor->in_, buffer, visitedTensors, visitedNodes);
  }
  */
}

void Flow::deduceIns(Node* node, std::vector<Tensor*>& buffer, std::set<Tensor*>& visitedTensors, std::set<Node*>& visitedNodes) const {
  /*
  if (visitedNodes.contains(node)) return;
  visitedNodes.insert(node);

//  std::cout << "ndoe: " << node << std::endl;
  for (auto& in: node->ins_) {
    deduceIns(in, buffer, visitedTensors, visitedNodes);
  }
  */
}

void Flow::prune(Out* out) {

}

}
}
