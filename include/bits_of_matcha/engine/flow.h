#pragma once

#include "bits_of_matcha/engine/object.h"
#include "bits_of_matcha/tuple.h"

#include <iostream>
#include <vector>
#include <set>


namespace boost {
namespace serialization {
  class access;
}
}

namespace matcha {

class Tensor;
class Params;
class Stream;
class Input;
class Tuple;
class Flow;

namespace engine {

class Tensor;
class Stream;
class Params;
class Input;
class Node;


class Flow : public Object {
  public:
    Flow(const matcha::Tuple& outs);

    void save(std::ostream& os) const;
    static matcha::Flow load(std::istream& is);

    const matcha::Tensor& in(int index) const;
    const matcha::Tensor& out(int index) const;

    size_t ins() const;
    size_t outs() const;

  public:
    void considerPruning() override;

  private:
    mutable Tuple outs_;
    mutable Tuple ins_;

    mutable std::vector<Tensor*> logic_;
    mutable std::vector<Params*> params_;
    mutable std::vector<Stream*> streams_;

  private:
    void deduceIns() const;
    void deduceIns(Tensor* tensor, std::vector<Tensor*>& buffer, std::set<Tensor*>& visitedTensors, std::set<Node*>& visistedNodes) const;
    void deduceIns(Node* node, std::vector<Tensor*>& buffer, std::set<Tensor*>& visitedTensors, std::set<Node*>& visitedNodes) const;

    friend class FlowSaver;
};


}
}
