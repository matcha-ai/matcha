#include "bits_of_matcha/engine/utils/TexGraph.h"
#include "bits_of_matcha/engine/flow/graph/TensorMask.h"
#include "bits_of_matcha/engine/flow/graph/OpMask.h"
#include "bits_of_matcha/print.h"


namespace matcha::engine::utils {

void TexGraph::run(std::ostream& os) {
  if (claimGraphCtx) {
    auto ctx = graph->ctx();
    dump(os);
  } else {
    dump(os);
  }
}

void TexGraph::dump(std::ostream& os) {
  auto opNodes = getOpNodes();
  auto tensorNodes = getTensorNodes();
  auto paths = getPaths(opNodes, tensorNodes);
  space(opNodes, tensorNodes);

  if (includeHeader) dumpHeader(os);
  os << "\\begin{figure}\n";
  os << "\\centering\n";
  os << "\\begin{tikzpicture}\n";

  for (auto node: opNodes) if (node) node->dump(os);
  for (auto node: tensorNodes) if (node) node->dump(os);
  for (auto path: paths) if (path) path->dump(os);

  os << "\\end{tikzpicture}\n";
  if (!caption.empty()) {
    os << "\\caption{" << caption << "}\n";
  }
  os << "\\end{figure}\n";
  if (includeFooter) dumpFooter(os);
  std::flush(os);
}

OpDict<TexGraph::TikzNode*> TexGraph::getOpNodes() {
  auto opDepths = getOpDepths();
  int depth = *std::max_element(opDepths.begin(), opDepths.end());
  OpDict<TikzNode*> tikz(graph, nullptr);
  for (auto op: graph->ops) {
    auto defaultInfo = defaultOpInfo(op);
    auto info = opInfo(op);

    if (info.label.empty()) info.label = defaultInfo.label;
    if (info.color.empty()) info.color = defaultInfo.color;

    tikz[op] = new TikzNode {
      .id = "node" + std::to_string(op->ctx().key()),
      .label = info.label,
      .style = "circle,minimum size=6mm,text width=1,align=center,fill=" + info.color,
    };
  }

  return tikz;
}

TensorDict<TexGraph::TikzNode*> TexGraph::getTensorNodes() {
  TensorDict<TikzNode*> tikz(graph, nullptr);
  TensorDict<int> deps(graph);
  TensorDict<Op*> sources(graph, nullptr);
  TensorMask ins(graph);
  TensorMask outs(graph);
  for (auto in: graph->inputs) ins[in] = true;
  for (auto out: graph->outputs) outs[out] = true;

  OpMask visited(graph);

  for (auto op: graph->ops) {
    graph->dfsPostfix(
      op,
      [&] (Op* op) {
        for (auto in: op->inputs) {
          if (!in) continue;
          deps[in]++;
        }
        for (auto out: op->outputs) {
          if (!out) continue;
          sources[out] = op;
        }
      },
      visited
    );
  }


  int total = 0;
  for (auto tensor: graph->tensors) {
    if (tensor->op() && deps[tensor] == 1) continue;
    auto defaultInfo = defaultTensorInfo(tensor);
    auto info = tensorInfo(tensor);

    if (info.label.empty()) info.label = defaultInfo.label;
    if (info.color.empty()) info.color = defaultInfo.color;

    tikz[tensor] = new TikzNode {
      .id = "tensor" + std::to_string(tensor->ctx().key()),
      .label = info.label,
      .style = "rectangle,fill=" + info.color,
    };
    total++;
  }

  return tikz;
}

std::vector<TexGraph::TikzPath*> TexGraph::getPaths(const OpDict<TikzNode*>& opNodes, const TensorDict<TikzNode*>& tensorNodes) {
  std::vector<TikzPath*> tikz;
  std::vector<int> deps(graph->tensors.size(), 0);
  OpMask visitedOps(graph);
  TensorMask visistedTensors(graph);

  size_t minSize = -1;
  size_t maxSize = 0;

  for (auto tensor: graph->tensors) {
    size_t size = tensor->size();
    if (size > maxSize) maxSize = size;
    if (size < minSize) minSize = size;
  }

  for (auto op: graph->ops) {
    graph->dfsPostfix(
    op,
    [&](Op* op) {
      for (auto in: op->inputs) {
        auto tensorNode = tensorNodes[in];
        auto defaultInfo = defaultTensorInfo(in);
        auto info = tensorInfo(in);
        TikzPath* path;

        if (info.label.empty()) info.label = defaultInfo.label;
        if (info.color.empty()) info.color = defaultInfo.color;

        if (tensorNode) {
          path = new TikzPath {
            .a = tensorNode,
            .b = opNodes[op],
            .label = "",
            .style = info.color,
          };
        } else {
          path = new TikzPath {
            .a = opNodes[in->op()],
            .b = opNodes[op],
            .label = "",
            .style = info.color,
          };
        }
        tikz.push_back(path);
      }
    },
    visitedOps
    );
  }

  for (auto tensor: graph->tensors) {
    auto op = tensor->op();
    auto tensorNode = tensorNodes[tensor];
    if (!op || !tensorNode) continue;

    auto defaultInfo = defaultTensorInfo(tensor);
    auto info = tensorInfo(tensor);

    if (info.label.empty()) info.label = defaultInfo.label;
    if (info.color.empty()) info.color = defaultInfo.color;

    auto path = new TikzPath {
      .a = opNodes[op],
      .b = tensorNode,
      .label = info.label,
      .style = info.color,
    };

    tikz.push_back(path);
  }

  return tikz;
}

void TexGraph::TikzNode::dump(std::ostream& os) {
  os << "\t\\node[font=\\sffamily," << style << "] ("
     << id
     << ") at (" << x << "," << y << ") {"
     << label
     << "};\n";
}

void TexGraph::TikzPath::dump(std::ostream& os) {
  os << "\t\\draw[->,thick," << style << "] "
     << "(" << a->id << ") -- (" << b->id << ");\n";
}

void TexGraph::dumpHeader(std::ostream& os) {
  os << "\\documentclass{article}\n"
     << "\\usepackage{tikz}\n"
     << "\\usetikzlibrary{arrows.meta}\n"
     << "\\begin{document}\n";
}

void TexGraph::dumpFooter(std::ostream& os) {
  os << "\\end{document}\n";
}

std::ostream& operator<<(std::ostream& os, TexGraph texGraph) {
  texGraph.run(os);
  return os;
}

OpDict<int> TexGraph::getOpDepths() {
  OpDict<int> depths(graph, -1);
  for (auto op: graph->ops) {
    OpDict<int> tempDepths(graph, -1);
    graph->dfsPostfix(op, [&](auto){}, tempDepths);
    std::transform(
      depths.begin(), depths.end(),
      tempDepths.begin(),
      depths.begin(),
      [](int a, int b) { return std::max(a, b); }
    );
  }
  return depths;
}

TensorDict<int> TexGraph::getTensorDepths(const OpDict<int>& opDepths) {
  TensorDict<int> depths(graph);
  for (auto op: graph->ops) {
    for (auto in: op->inputs) {
      depths[in] = std::max(depths[in], opDepths[op] + 1);
    }
  }

  for (auto out: graph->outputs) {
    depths[out] = 0;
  }

  return depths;
}

void TexGraph::space(OpDict<TikzNode*>& opNodes, TensorDict<TikzNode*>& tensorNodes) {
  auto opDepths = getOpDepths();
  auto tensorDepths = getTensorDepths(opDepths);
  auto graphDepth = *std::max_element(opDepths.begin(), opDepths.end());
  if (graph->ops.empty()) graphDepth = 1;

  std::vector<int> horizontalOpSpacing(graphDepth + 1, 0);
  auto opNoise = [cycle = 6](Op* op) {
    int iter = op->ctx().key() % cycle;
    if (iter >= cycle / 2) iter = cycle - iter;
    return iter;
  };

  for (auto op: graph->ops) {
    auto& node = *opNodes[op];
    int depth = opDepths[op];
    node.y = (graphDepth - depth) * 2.2;
    node.x = horizontalOpSpacing[depth] * 2 + opNoise(op) * 1;
    horizontalOpSpacing[depth]++;
  }

  std::vector horizontalTensorSpacing(graphDepth + 2, 0);
  auto tensorNoise = [cycle = 6](Tensor* tensor) {
    int iter = tensor->ctx().key() % cycle;
    if (iter >= cycle / 2) iter = cycle - iter;
    return tensor->ctx().key() % 15;
  };

  for (auto tensor: graph->tensors) {
    if (!tensorNodes[tensor]) continue;
    auto& node = *tensorNodes[tensor];
    auto op = tensor->op();
    int depth = tensorDepths[tensor];
    node.y = (graphDepth - depth) * 2.2 + 1.1;

//    if (op) {
//      node.x = opNodes[op]->x;
//    } else {
      node.x = horizontalTensorSpacing[depth] * 2 + tensorNoise(tensor) * .2;
//    }
    horizontalTensorSpacing[depth]++;
  }
}

}