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

class Ops {
public:
  Ops() = delete;

  struct Entry {
    std::function<std::string ()> name;
    std::function<std::string (Op* op)> label;
    std::function<Op* (const BackCtx&)> back;
    std::function<bool (Op* op)> effect;

    ~Entry();
  };
  static void add(std::type_index op, Entry* entry);

  static std::string name(Op* op);
  static std::string label(Op* op);
  static Op* back(const BackCtx& ctx);

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
Op* back(const BackCtx& ctx);

}
