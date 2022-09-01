#include "bits_of_matcha/engine/lambda/passes/debug.h"
#include "bits_of_matcha/engine/lambda/passes/check.h"
#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/utils/IdentifierTranslator.h"

#include <map>

namespace matcha::engine {

void debug(const Lambda& lambda, std::ostream& os) {
  os << std::string(30, '=') << " CHAIN CHECK BEGIN " << std::string(30, '=') << std::endl;
  os << std::endl;
  IdentifierTranslator<Tensor*> it;

  os << "lambda(";
  for (int i = 0; i < lambda.inputs.size(); i++) {
    auto in = lambda.inputs[i];
    if (i != 0) os << ", ";
    os << it(in) << ": " << in->frame();
  }
  os << ")";

  for (int i = 0; i < lambda.outputs.size(); i++) {
    if (i == 0) os << " -> ";
    else os << ", ";
    os << lambda.outputs[i]->frame();
  }

  os << " {" << std::endl;

  for (auto&& op: lambda.ops) {
    for (auto&& in: op->inputs) it(in);
    for (auto&& out: op->outputs) it(out);

    os << std::string(4, ' ');

    for (int i = 0; i < op->outputs.size(); i++) {
      if (i != 0) os << ", ";
      os << it(op->outputs[i]);
    }

    if (!op->outputs.empty()) os << " = ";

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

  if (!lambda.outputs.empty()) {
    if (!lambda.ops.empty()) os << std::endl;
    os << std::string(4, ' ') << "return ";
    for (int i = 0; i < lambda.outputs.size(); i++) {
      if (i != 0) os << ", ";
      os << it(lambda.outputs[i]);
    }
    os << std::endl;
  }

  os << "}" << std::endl;

  os << std::endl;

  for (auto&& t: lambda.tensors) {
    os << it(t) << " \t";
//    os << "[refs,reqs: " << t->refs() << "," << t->reqs() << "] \t";
    os << "op: " << t->op();
    if (t->op()) os << " (" << ops::name(t->op()) << ")";

    size_t sins = 0;
    for (auto&& sin: lambda.side_inputs) {
      if (sin.first != t) continue;
      if (sins++ == 0) {
        os << "\t side in:";
      } else {
      }
      os << " ";
      os << sin.second;
      os << " (" << sin.second->frame() << ")";
    }

    os << std::endl;
  }

  os << std::endl;

  int validity = check(lambda);
  if (validity != 0) {
    os << "WARNING: lambda validity check returned with non-zero status: "
       << validity << "\n\n";
  }

  os << std::string(30, '=') << "= CHAIN CHECK END =" << std::string(30, '=') << std::endl;
}

}