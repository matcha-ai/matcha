#include "bits_of_matcha/engine/nodeserializer.h"
#include "bits_of_matcha/engine/flowsaver.h"
#include "bits_of_matcha/engine/node.h"


namespace matcha {
namespace engine {


NodeSerializer::NodeSerializer(std::initializer_list<std::function<const NodeLoader* ()>> loaders) {
  for (auto& loader: loaders) {
    auto* l = loader();
    defaultRegister_[l->type] = l;
  }
}

Node* NodeSerializer::load(std::istream& is, const std::string& type, const std::vector<Tensor*>& ins) const {
  if (defaultRegister_.contains(type)) {
    return defaultRegister_.at(type)->load(is, ins);
  } else if (extendedRegister_.contains(type)) {
    return extendedRegister_.at(type)->load(is, ins);
  } else {
    throw std::invalid_argument("node type \"" + type + "\" is not registered in the engine");
  }
}

void NodeSerializer::save(std::ostream& os, Node* node) const {
  if (node->polymorphicOuts()) {
    savePolymorphic(os, node);
  } else {
    saveNonPolymorphic(os, node);
  }
}

void NodeSerializer::addLoader(const NodeLoader* loader) {
  extendedRegister_[loader->type] = loader;
}

void NodeSerializer::savePolymorphic(std::ostream& os, Node* node) const {
  auto loader = node->getLoader();
  if (loader == nullptr) throw std::runtime_error("Node Loader is null");

  os << "\n";
  for (auto& out: node->outs_) {
    FlowSaver::declareTensor(os, out);
  }
  os << "\n";

  FlowSaver::assignment(os, "*" + node->name(), " = ");
  os << loader->type;
  FlowSaver::saveIns(os, node->ins_);
  os << " {";

  node->save(os);

  os << "\n}\n\n";
  for (auto& out: node->outs_) {
    FlowSaver::connection(os, node, {out});
  }
  os << "\n\n";

}

void NodeSerializer::saveNonPolymorphic(std::ostream& os, Node* node) const {
  auto loader = node->getLoader();
  if (loader == nullptr) throw std::runtime_error("Node Loader is null");

  for (int i = 0; i < node->outs(); i++) {
    if (i != 0) os << ", ";
    os << node->out(i)->name();
  }
  os << " = ";

  os << loader->type;
  FlowSaver::saveIns(os, node->ins_);
  os << " {";

  node->save(os);

  os << "\n}\n\n";

}

}
}
