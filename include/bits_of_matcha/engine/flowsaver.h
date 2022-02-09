#pragma once

#include "bits_of_matcha/engine/object.h"
#include "bits_of_matcha/dtype.h"
#include "bits_of_matcha/shape.h"

#include <iostream>
#include <set>
#include <queue>
#include <map>


namespace matcha {
namespace engine {

class Flow;
class Node;
class Tensor;


class FlowSaver {
  public:
    static void save(std::ostream& os, const Flow* flow);

    static void declareTensor(std::ostream& os, Tensor* tensor);
    static void indent(std::ostream& os);
    static void tuple(std::ostream& os, const std::vector<std::string>& names);
    static void tuple(std::ostream& os, const std::vector<Object*>& names);
    static void assignment(std::ostream& os, const std::vector<std::string>& lvalues, const std::string& assignmentSymbol);
    static void assignment(std::ostream& os, const std::string& lvalue, const std::string& assignmentSymbol);
    static void connection(std::ostream& os, const std::string& in, const std::vector<std::string>& outs);
    static void connection(std::ostream& os, Node* in, const std::vector<Tensor*>& outs);
    static void dtype(std::ostream& os, const Dtype& dtype);
    static void shape(std::ostream& os, const Shape& shape);
    static void saveIns(std::ostream& os, const std::vector<Tensor*>& ins);
    static void oneFloat(std::ostream& os, float value);
    static void flatFloats(std::ostream& os, const float* buffer, const Shape& shape = {}, unsigned indent = 0);

  private:
    FlowSaver(std::ostream& os, const Flow* flow);

    void schedule(Node* node);
    void schedule(Tensor* tensor);

  private:
    std::set<Tensor*> scheduledTensors_;
    std::set<Node*> scheduledNodes_;
    std::queue<Tensor*> tensorQueue_;
    std::queue<Node*> nodeQueue_;

  private:
    std::ostream& os_;
    const Flow* flow_;
};


}
}
