#pragma once

#include <typeindex>
#include <functional>
#include <string>
#include <unordered_map>
#include <memory>

namespace matcha::engine {

class Op;
class OpBack;
class BackCtx;
class BackOps;

class Ops final {
public:
  Ops() = delete;

  struct Entry {
    std::function<std::string ()> name;
    std::function<std::string (Op* op)> label;
    std::function<BackOps (const BackCtx&)> back;
    std::function<bool (Op* op)> sideEffect;
    std::function<Op* (Op* op)> copy;

    ~Entry();
  };
  static void add(std::type_index op, Entry* entry);

  static std::string name(Op* op);
  static std::string label(Op* op);
  static BackOps back(const BackCtx& ctx);
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
BackOps back(const BackCtx& ctx);
bool isSideEffect(Op* op);
Op* copy(Op* op);

}
