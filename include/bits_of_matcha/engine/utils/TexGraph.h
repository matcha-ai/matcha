#pragma once

#include "bits_of_matcha/engine/flow/graph/Graph.h"
#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"

#include <iostream>


namespace matcha::engine::utils {

struct TexGraph {
public:
  void run(std::ostream& os);

  struct OpInfo {
    std::string label;
    std::string color;
  };

  struct TensorInfo {
    std::string label;
    std::string color;
  };

  struct TikzNode {
    std::string id;
    std::string label = id;
    std::string style;
    float x = 0;
    float y = 0;

    void dump(std::ostream& os);
  };

  struct TikzPath {
    TikzNode* a;
    TikzNode* b;
    std::string label;
    std::string style;

    void dump(std::ostream& os);
  };

  Graph* graph;
  bool claimGraphCtx = true;

  std::string caption = "\\texttt{matcha::Flow}";
  std::tuple<float, float> dims = {11.0, 19.7}; // A4 with margins

  std::function<OpInfo (Op*)> defaultOpInfo = [](Op* op) {
    return OpInfo {
      .label = ops::label(op),
      .color = "blue!20",
    };
  };

  std::function<TensorInfo (Tensor*)> defaultTensorInfo = [&](Tensor* tensor) {
    std::string label = tensor->frame().string();
    std::string color = "blue!20";

    auto& ins = graph->inputs;
    auto& outs = graph->outputs;

    for (int i = 0; i < ins.size(); i++) {
      auto in = ins[i];
      if (tensor != in) continue;

      label += " $i_" + std::to_string(i) + "$";
      color  = "purple";
    }

    for (int i = 0; i < outs.size(); i++) {
      auto out = outs[i];
      if (tensor != out) continue;

      label += " $o_" + std::to_string(i) + "$";
      color  = "purple";
    }

    return TensorInfo {
      .label = label,
      .color = color
    };
  };

  std::function<OpInfo (Op*)> opInfo = [](auto) { return OpInfo{}; };
  std::function<TensorInfo (Tensor*)> tensorInfo = [](auto) { return TensorInfo{}; };

  bool includeHeader = true;
  bool includeFooter = true;

private:
  void dumpHeader(std::ostream& os);
  void dumpFooter(std::ostream& os);

  void space(OpDict<TikzNode*>& opNodes, TensorDict<TikzNode*>& tensorNodes);
  OpDict<int> getOpDepths();
  OpDict<TikzNode*> getOpNodes();
  TensorDict<int> getTensorDepths(const OpDict<int>& opDepths);
  TensorDict<TikzNode*> getTensorNodes();
  std::vector<TikzPath*> getPaths(const OpDict<TikzNode*>& opNodes, const TensorDict<TikzNode*>& tensorNodes);
  std::tuple<float, float> getContentDims(const OpDict<TikzNode*>& opNodes, const TensorDict<TikzNode*>& tensorNodes);
  std::tuple<float, float> getGridDims(const std::tuple<float, float>& content, const std::tuple<float, float>& target);
  void dump(std::ostream& os);

};

std::ostream& operator<<(std::ostream& os, TexGraph texGraph);

}
