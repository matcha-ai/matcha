#include "bits_of_matcha/engine/chain/Chain.h"
#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/ops/Assign.h"

#include <map>

namespace matcha::engine {

Chain::~Chain() {
//  std::cerr << "~Chain " << ops.size() << std::endl;
  for (auto&& op: ops) {
//    std::cerr << "deleting " << op << " " << ops::name(op) << std::endl;
    delete op;
  }
  for (auto&& t: tensors) {
//    std::cerr << t << " " << t->reqs() << std::endl;
//    if (t->op()) std::cerr << ops::name(t->op()) << std::endl;
//    std::cerr << t << std::endl;
    t->unreq();
  }
}

Chain copy(const Chain& chain) {
  std::map<Tensor*, Tensor*> tensors;
  Chain result;

  for (auto&& in: chain.inputs) {
    tensors[in] = new Tensor(in->frame());
  }

  auto relink = [&] (Tensor* t) {
    if (tensors.contains(t)) return tensors.at(t);
    if (!t->op()) return t;
    auto t2 = new Tensor(t->frame());
    tensors[t] = t2;
    return t2;
  };

  for (auto&& op: chain.ops) {
    auto cop = ops::copy(op);
    if (!cop || cop == op) throw std::runtime_error("can't copy op");
    result.ops.push_back(cop);

    for (auto& in: cop->inputs) {
      if (!in) continue;
      in = relink(in);
      in->req();
    }
    for (auto& out: cop->outputs) {
      if (!out) continue;
      out = relink(out);
      out->setOp(cop);
    }
  }
  for (auto&& t: chain.tensors) result.tensors.push_back(relink(t));
  for (auto&& in: chain.inputs) result.inputs.push_back(relink(in));
  for (auto&& out: chain.outputs) result.outputs.push_back(relink(out));

  for (auto&& t: result.tensors) t->req();

  return result;
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
//      os << out;
    }

    if (op->outputs.any())
      os << " = ";

    os << opname << '(';

    for (int i = 0; i < op->inputs.size(); i++) {
      auto in = op->inputs[i];
      if (i != 0) os << ", ";
      os << tid(in);
//      os << in;
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