#include "bits_of_matcha/engine/node.h"
#include "bits_of_matcha/engine/tensor.h"
#include "bits_of_matcha/tensor.h"

#include <stdexcept>
#include <algorithm>


namespace matcha {
namespace engine {

Node::Node(std::initializer_list<Tensor*> ins)
{
  for (auto& in: ins) {
    ins_.push_back(createIn(in->out(), ins_.size()));
  }

  bool data = std::all_of(
    std::begin(ins_), std::end(ins_),
    [](auto* in) {
      return in->status().data;
    }
  );

  status_ = {
    .data = data,
    .update = true,
    .ready = false
  };
}

In* Node::in(int index) {
  if (index < 0) index += ins();
  if (index < 0 || index >= ins()) throw std::out_of_range("node in index is out of range");
  return ins_[index];
}

size_t Node::ins() const {
  return ins_.size();
}

Out* Node::out(int index) {
  if (index < 0) index += outs();
  if (index < 0 || index >= outs()) throw std::out_of_range("node out index is out of range");
  return outs_[index];
}

size_t Node::outs() const {
  return outs_.size();
}

Tensor* Node::deref(const matcha::Tensor* tensor) {
  return tensor->object();
}

Tensor* Node::deref(const matcha::Tensor& tensor) {
  return tensor.object();
}

void Node::dataStatusChanged(In *in) {
//  for (auto* out: outsNew_) out->dataStatusChanged(data);
}

void Node::updateStatusChanged(In *in) {
  for (auto* out: outs_) out->updateStatusChanged();
}


}
}
