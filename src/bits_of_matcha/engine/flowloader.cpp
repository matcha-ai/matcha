#include "bits_of_matcha/engine/flowloader.h"
#include "bits_of_matcha/engine/nodeserializer.h"
#include "bits_of_matcha/engine/flow.h"
#include "bits_of_matcha/tensor.h"

#include <sstream>


namespace matcha {
namespace engine {


Flow* FlowLoader::load(std::istream& is) {
  auto fl = FlowLoader(is);
  return fl.get();
}

FlowLoader::FlowLoader(std::istream& is) {
  using namespace std;
  string symbol;
  string version;

  is >> symbol;
  if (symbol == "@matcha") {
    is >> version;
  } else {
    version = "0.0.0";
    is.seekg(0);
  }

  initKeywords();

  while (true) {
    ignoreWhitespaces(is);
    if (is.eof()) break;

    if (tryKeyword(is)) continue;
    if (tryRedirect(is)) continue;

    auto lvals = lvalues(is);
    string type = getType(is);
    bool isTensor;

    try {
      Dtype dtype(type);
      isTensor = true;
    } catch (std::invalid_argument& e) {
      isTensor = false;
    }

    if (isTensor) {
      auto [dtype, shape] = parseTensor(type, is);
      for (auto& name: lvals) {
        addTensor(name, new Tensor(dtype, shape));
      }
    } else {
      /*
      Node* node = parseNode(type, is);

      if (node->polymorphicOuts()) {
        if (lvals.size() != 1) throw std::invalid_argument("semantic error: the number of lvalues for polymorphic node must be exactly one");
        string& lval = lvals[0];
        if (lval[0] != '*') throw std::invalid_argument("syntax error: node name must immediately follow *");
        addNode(&lval[1], node);
      } else {
        if (lvals.size() != node->outs()) {
          throw std::invalid_argument("semantic error: the number of lvalues doesn't match the number of rvalues");
        }
        for (int i = 0; i < lvals.size(); i++) {
          addTensor(lvals[i], node->out(i));
        }
      }
      */
    }
  }

}

bool FlowLoader::ignoreComment(std::istream& is, bool initialConsumed) {
  using namespace std;

  auto begin = is.tellg();
  if (!initialConsumed) {
    // /
    is.get();
  }

  char c = is.peek();
  string buffer;
  if (c == '/') {
    // "// comment"
    getline(is, buffer);
    return true;
  }

  if (c != '*') {
    is.seekg(begin);
    return false;
  }

  // "/* comment */"
  is.get();
  while (!is.eof()) {
    c = is.get();
    if (c == '*' && is.get() == '/') break;
  }

  return true;
}

void FlowLoader::ignoreWhitespaces(std::istream& is, bool allowNewline) {
  using namespace std;

  if (!allowNewline) {
    while (::isspace(is.peek())) is.get();
    return;
  }

  while (!is.eof()) {
    char c = is.peek();
    if (c == '/') {
      if (ignoreComment(is, false)) continue;
      else break;
    }
    if (!isspace(c)) break;
    is.get();
  }
}

void FlowLoader::initKeywords() {
  using namespace std;

  keywords_["yield"] = [&](auto& is) {
    static bool set = false;
    if (set) {
      throw std::invalid_argument("semantic error: yield can be defined only once");
    }
    set = true;

    string line;
    getline(is, line);
    stringstream ss(line);
    auto outs = tuple(ss);
    setOuts(getTensors(outs));
  };

}

bool FlowLoader::tryKeyword(std::istream& is) {
  using namespace std;
  ignoreWhitespaces(is);

  auto pos = is.tellg();
  string buffer;
  is >> buffer;

  std::function<void (std::istream& is)>* action;
  try {
    action = &keywords_.at(buffer);
  } catch (...) {
    is.seekg(pos);
    return false;
  }

  (*action)(is);
  return true;
}

bool FlowLoader::tryRedirect(std::istream& is) {
  using namespace std;
  ignoreWhitespaces(is);

  auto pos = is.tellg();
  auto lval = lvalue(is);

  string buffer;
  is >> buffer;
  if (buffer != "=>") {
    is.seekg(pos);
    return false;
  }

  getline(is, buffer);
  stringstream ss(buffer);
  auto rvals = tuple(ss);
  if (rvals.empty()) throw std::invalid_argument("syntax error: missing redirect target");
  if (lval[0] != '*') throw std::invalid_argument("semantic error: only nodes can be redirected");

  Node* node = getNode(&lval[1]);
  for (auto& rval: rvals) {
    Tensor* tensor = getTensor(rval);
//    node->openOut(tensor);
  }

  return true;
}

std::vector<std::string> FlowLoader::tuple(std::istream &is) {
  using namespace std;
  ignoreWhitespaces(is);

  vector<string> elems;

  std::string elemBuffer;
  while (true) {
    char c = is.get();
    if (c == EOF) break;
    switch (c) {
      case ',':
        if (elemBuffer.empty()) {
          throw std::invalid_argument("syntax error: missing an argument before ,");
        }
        elems.push_back(elemBuffer);
        elemBuffer.clear();
        ignoreWhitespaces(is);
        break;

      case ' ':
        ignoreWhitespaces(is);
        if (is.peek() != EOF) {
          throw std::invalid_argument("syntax error: missing ,");
        }
        elems.push_back(elemBuffer);
        elemBuffer.clear();
        break;

      case '\n':
        throw std::invalid_argument("syntax error: newline must be preceded by ,");

      default:
        if (elemBuffer.empty()) {
          if (::isdigit(c)) {
            throw std::invalid_argument("syntax error: variable can't begin with a number");
          }
          if (::isalpha(c) && ::toupper(c) == c) {
            throw std::invalid_argument("syntax error: variable can't begin with a capital");
          }
        }
        if (!::isalnum(c)) {
          switch (c) {
            case '_':
            case '.':
              break;
            case '*':
              if (elemBuffer.empty()) break;
            default:
              throw std::invalid_argument(
                std::string("syntax error: variable contains forbidden character: ") + c
              );
          }
        }

        elemBuffer += c;
    }
  }

  if (!elemBuffer.empty()) {
    elems.push_back(elemBuffer);
  }

  return elems;
}

std::vector<std::string> FlowLoader::lvalues(std::istream& is, char assignmentSymbol) {
  using namespace std;
  string buffer;
  getline(is, buffer, assignmentSymbol);
  stringstream ss(buffer);

  auto elems = tuple(ss);
  ignoreWhitespaces(is);
  return elems;
}

std::string FlowLoader::lvalue(std::istream& is, char assignmentSymbol) {
  auto lvals = lvalues(is, assignmentSymbol);
  if (lvals.size() != 1) throw std::invalid_argument("semantic error: expected exactly one lvalue");
  return lvals[0];
}

Dtype FlowLoader::dtype(std::istream& is) {
  return Dtype(getType(is));
}

Shape FlowLoader::shape(std::istream& is) {
  using namespace std;

  std::string buffer;
  if (is.get() != '[') throw std::invalid_argument("syntax error: missing [ before shape");
  getline(is, buffer, ']');
  stringstream ss(buffer);

  vector<unsigned> dims;
  string dimBuffer;

  while (ss.peek() != EOF) {
    getline(ss, dimBuffer, ',');
    try {
      int dim = std::stoi(dimBuffer);
      if (dim <= 0) throw;
      dims.push_back(dim);
    }  catch (...) {
      throw std::invalid_argument("syntax error: invalid dimension");
    }
  }
  return Shape(dims);
}

float FlowLoader::oneFloat(std::istream& is) {
  using namespace std;

  string buffer;
  is >> buffer;
  try {
    return stof(buffer);
  } catch (...) {
    throw std::invalid_argument("semantic error: expected Float, got some garbage instead");
  }
}

float FlowLoader::oneUint64(std::istream& is) {
  using namespace std;

  string buffer;
  is >> buffer;
  try {
    auto ll = stoll(buffer);
    if (ll < 0) throw;
    return ll;
  } catch (...) {
    throw std::invalid_argument("semantic error: expected Uint64, got some garbage instead");
  }
}


std::vector<float> FlowLoader::flatFloats(std::istream& is) {
  using namespace std;
  ignoreWhitespaces(is);

  if (is.get() != '{') throw std::invalid_argument("syntax error: missing { before flat data block");
  string buffer;
  getline(is, buffer, '}');
  stringstream ss(buffer);

  std::vector<float> data;

  while (ss.peek() != EOF) {
    data.push_back(oneFloat(ss));
    ignoreWhitespaces(ss);
  }
  return data;
}

std::string FlowLoader::getType(std::istream& is) {
  using namespace std;
  ignoreWhitespaces(is);

  string type;
  type.reserve(16);
  while (!is.eof()) {
    bool special = false;
    char c = is.peek();
    switch (c) {
      case '[':
      case '(':
      case '{':
        special = true;
        break;
      default:
        if (isspace(c)) {
          special = true;
        }
    }

    if (special) {
      break;
    } else {
      type += is.get();
    }
  }
  return type;
}

std::tuple<Dtype, Shape> FlowLoader::parseTensor(const std::string& type, std::istream& is) {
  using namespace std;

  if (is.get() != '[') throw std::invalid_argument("syntax error: expected shape");
  string buffer;
  getline(is, buffer, ']');
  stringstream ss(buffer);

  vector<unsigned> axes;

  string axisBuffer;
  while (ss >> axisBuffer) {
    try {
      int axis;
      axis = std::stoi(axisBuffer);
      axes.push_back(axis);
      if (axis < 0) throw std::invalid_argument("");
    } catch (std::invalid_argument& e) {
      throw std::invalid_argument("syntax error: invalid dimension");
    }
  }
  return {type, axes};
}

void FlowLoader::addTensor(const std::string& name, Tensor* tensor) {
  if (tensors_.contains(name) || nodes_.contains(name)) {
    throw std::invalid_argument("semantic error: redefinition of " + name);
  }
  tensors_.insert({name, tensor});
  tensor->rename(name);
}

void FlowLoader::addNode(const std::string& name, Node* node) {
  if (tensors_.contains(name) || nodes_.contains(name)) {
    throw std::invalid_argument("semantic error: redefinition of " + name);
  }
  nodes_.insert({name, node});
  node->rename(name);
}

Node* FlowLoader::parseNode(const std::string& type, std::istream& is) {
  using namespace std;

  auto ins = getIns(is);
  ignoreWhitespaces(is);

  if (is.get() != '{') {
    throw std::invalid_argument("syntax error: expected node properties");
  }

  ignoreWhitespaces(is);
  Node* node = nodeSerializer.load(is, type, ins);
  ignoreWhitespaces(is);

  if (is.get() != '}') {
    throw std::invalid_argument("syntax error: missing } after node properties");
  }

  return node;
}

std::vector<Tensor*> FlowLoader::getIns(std::istream& is) {
  using namespace std;

  if (is.get() != '(') {
    ignoreWhitespaces(is);
    if (is.peek() == '{') return {};
    throw std::invalid_argument("syntax error: exptected node arguments or properties");
  }

  string buffer;
  getline(is, buffer, ')');
  stringstream ss(buffer);

  auto args = tuple(ss);

  vector<Tensor*> ins;
  ins.reserve(args.size());
  std::transform(
    std::begin(args), std::end(args),
    std::back_inserter(ins),
    [&](auto& in) {
      try {
        return tensors_.at(in);
      } catch (...) {
        throw std::invalid_argument("semantic error: tensor " + in + " has not been declared");
      }
    }
  );

  return ins;
}

Tensor* FlowLoader::getTensor(const std::string& name) {
  try {
    return tensors_.at(name);
  } catch (...) {
    throw std::invalid_argument("semantic error: tensor " + name + " has not been declared");
  }
}

Node* FlowLoader::getNode(const std::string& name) {
  try {
    return nodes_.at(name);
  } catch (...) {
    throw std::invalid_argument("semantic error: tensor " + name + " has not been declared");
  }
}

std::vector<Tensor*> FlowLoader::getTensors(const std::vector<std::string>& names) {
  std::vector<Tensor*> tensors;
  tensors.reserve(names.size());
  std::transform(
    std::begin(names), std::end(names),
    std::back_inserter(tensors),
    [&](auto& name) {
      return getTensor(name);
    }
  );
  return tensors;
}

void FlowLoader::setOuts(const std::vector<Tensor *>& outs) {
  outs_ = outs;
}

Flow* FlowLoader::get() {
  using namespace std;
  using matcha::Tensor;

  vector<Tensor> tensors;
  tensors.reserve(outs_.size());
  std::transform(
    std::begin(outs_), std::end(outs_),
    std::back_inserter(tensors),
    [](auto tensor) {
      return Tensor::fromObject(tensor);
    }
  );

  return new Flow(tensors);
}


}
}
