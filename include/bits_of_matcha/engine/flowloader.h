#pragma once

#include "bits_of_matcha/engine/object.h"
#include "bits_of_matcha/dtype.h"
#include "bits_of_matcha/shape.h"


#include <functional>
#include <iostream>
#include <vector>
#include <map>


namespace matcha {

class Dtype;

namespace engine {

class Flow;
class Tensor;
class Node;


class FlowLoader {
  public:
    static Flow* load(std::istream& is);

    static std::vector<std::string> tuple(std::istream& is);
    static std::vector<std::string> lvalues(std::istream& is, char assignmentSymbol = '=');
    static std::string lvalue(std::istream& is, char assignmentSymbol = '=');

    static float oneFloat(std::istream& is);
    static float oneUint64(std::istream& is);

    static std::vector<float> flatFloats(std::istream& is);

    static Dtype dtype(std::istream& is);
    static Shape shape(std::istream& is);

  private:
    FlowLoader(std::istream& is);
    Flow* get();

  private:
    static bool ignoreComment(std::istream& is, bool initialConsumed);
    static void ignoreWhitespaces(std::istream& is, bool allowNewline = true);

    static std::string getType(std::istream& is);
    std::vector<Tensor*> getIns(std::istream& is);
    std::tuple<Dtype, Shape> parseTensor(const std::string& type, std::istream& is);
    Node* parseNode(const std::string& type, std::istream& is);

    std::map<std::string, Tensor*> tensors_;
    std::map<std::string, Node*> nodes_;

    void addTensor(const std::string& name, Tensor* tensor);
    void addNode(const std::string& name, Node* node);
    Tensor* getTensor(const std::string& name);
    Node* getNode(const std::string& name);

    std::vector<Tensor*> getTensors(const std::vector<std::string>& names);

    void initKeywords();
    bool tryKeyword(std::istream& is);
    bool tryRedirect(std::istream& is);
    std::map<std::string, std::function<void (std::istream& is)>> keywords_;

    void setOuts(const std::vector<Tensor*>& outs);
    std::vector<Tensor*> outs_;
};


}
}
