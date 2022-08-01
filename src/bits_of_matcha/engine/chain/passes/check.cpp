#include "bits_of_matcha/engine/chain/passes/check.h"
#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/utils/IdentifierTranslator.h"

#include <map>

namespace matcha::engine {

void check(const Chain& chain, std::ostream& os) {
  os << std::string(30, '=') << " CHAIN CHECK BEGIN " << std::string(30, '=') << std::endl;
  os << std::endl;
  IdentifierTranslator<Tensor*> it;

  os << "chain(";
  for (int i = 0; i < chain.inputs.size(); i++) {
    auto in = chain.inputs[i];
    if (i != 0) os << ", ";
    os << it(in) << ": " << in->frame();
  }
  os << ")";

  for (int i = 0; i < chain.outputs.size(); i++) {
    if (i == 0) os << " -> ";
    else os << ", ";
    os << chain.outputs[i]->frame();
  }

  os << " {" << std::endl;

  for (auto&& op: chain.ops) {
    for (auto&& in: op->inputs) it(in);
    for (auto&& out: op->outputs) it(out);

    os << std::string(4, ' ');

    for (int i = 0; i < op->outputs.size(); i++) {
      if (i != 0) os << ", ";
      os << it(op->outputs[i]);
    }

    if (op->outputs.any()) os << " = ";

    std::string opname;
    try {
      opname = ops::name(op);
    } catch (...) {
      opname = "Unknown";
    }

    os << opname << "(";
    for (int i = 0; i < op->inputs.size(); i++) {
      if (i != 0) os << ", ";
      os << it(op->inputs[i]);
    }
    os << ")" << std::endl;
  }

  if (!chain.outputs.empty()) {
    if (!chain.ops.empty()) os << std::endl;
    os << std::string(4, ' ') << "return ";
    for (int i = 0; i < chain.outputs.size(); i++) {
      if (i != 0) os << ", ";
      os << it(chain.outputs[i]);
    }
    os << std::endl;
  }

  os << "}" << std::endl;

  os << std::endl;

  for (auto&& t: chain.tensors) {
    os << it(t) << " \t" << "[refs,reqs: " << t->refs() << "," << t->reqs() << "] \t";
    os << "op: " << t->op();
    if (t->op()) os << " (" << ops::name(t->op()) << ")";
    os << std::endl;
  }

  os << std::endl;
  os << std::string(30, '=') << "= CHAIN CHECK END =" << std::string(30, '=') << std::endl;
}

}