#include "bits_of_matcha/engine/chain/Chain.h"
#include "bits_of_matcha/engine/op/Op.h"


namespace matcha::engine {

Chain::~Chain() {
  for (auto op: ops) delete op;
  for (auto t: tensors) t->unreq();
}

std::ostream& operator<<(std::ostream& os, const Chain& chain) {
  const int indent = 4;
  std::map<Tensor*, std::string> ids;
  const auto tid = [&] (Tensor* t) {

    if (ids.contains(t)) return ids.at(t);

    std::string id;
    size_t n = ids.size();
    size_t alphabetSize = 'z' - 'a' + 1;

    do {
      char c = n % alphabetSize + 'a';
      n /= alphabetSize;
      id += c;
    } while (n != 0);

    ids[t] = id;
    return id;
  };

  os << "chain(";
  for (int i = 0; i < chain.inputs.size(); i++) {
    auto in = chain.inputs[i];
    if (i != 0) os << ", ";
    os << tid(in) << ": " << in->frame();
  }
  os << ") -> ";

  for (int i = 0; i < chain.outputs.size(); i++) {
    auto out = chain.outputs[i];
    if (i != 0) os << ", ";
    os << out->frame();
  }
  os << " {\n";

  for (auto&& op: chain.ops) {
    std::string opname;
    try {
      opname = ops::name(op);
    } catch (...) {
      opname = "Unknown";
    }

    os << std::string(indent, ' ');

    for (int i = 0; i < op->outputs.size(); i++) {
      auto out = op->outputs[i];
      if (i != 0) os << ", ";
      os << tid(out);
    }

    if (op->outputs.any())
      os << " = ";

    os << opname << '(';

    for (int i = 0; i < op->inputs.size(); i++) {
      auto in = op->inputs[i];
      if (i != 0) os << ", ";
      os << tid(in);
    }
    os << ")\n";
  }

  os << "\n" << std::string(indent, ' ') << "return ";
  for (int i = 0; i < chain.outputs.size(); i++) {
    auto out = chain.outputs[i];
    if (i != 0) os << ", ";
    os << tid(out);
  }
  os << "\n}\n";

  return os;
}

}