#pragma once

#include <typeindex>
#include <functional>
#include <string>
#include <unordered_map>
#include <memory>

namespace matcha::engine {

class Op;
class BackCtx;
class Lambda;
class Tensor;

class Registry final {
public:
  Registry() = delete;

  struct Entry {
    std::function<std::string ()> name;
    std::function<std::string (Op* op)> label;
    std::function<std::vector<Tensor*> (const BackCtx&)> back;
    std::function<bool (Op* op)> side_effect;
    std::function<Op* (Op* op)> copy;

    ~Entry();
  };
  static void add(std::type_index op, Entry* entry);

  static std::string name(Op* op);
  static std::string label(Op* op);
  static Lambda back(const BackCtx& ctx);
  static bool isSideEffect(Op* op);
  static Op* copy(Op* op);

private:
  static Entry& get(Op* op);
  static Entry& get(const std::string& name);

  static std::unordered_map<std::type_index, Entry*>& entriesByType();
  static std::unordered_map<std::string, Entry*>& entriesByName();
};

}

namespace matcha::engine::ops {

std::string name(Op* op);
std::string label(Op* op);
Lambda back(const BackCtx& ctx);
bool isSideEffect(Op* op);
Op* copy(Op* op);

}
