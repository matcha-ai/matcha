#include "bits_of_matcha/engine/op/Ops.h"
#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/print.h"


namespace matcha::engine {

std::unordered_map<std::type_index, Ops::Entry*>& Ops::entriesByType() {
  static std::unordered_map<std::type_index, Entry*> entriesByType_ = {};
  return entriesByType_;
}

std::unordered_map<std::string, Ops::Entry*>& Ops::entriesByName() {
  static std::unordered_map<std::string, Entry*> entriesByName_;
  return entriesByName_;
}

void Ops::add(std::type_index op, Entry* entry) {
//  print("Registered Op: ", entry->name(), " with hash ", op.hash_code());
//  print(entriesByName().size());
  std::string name = entry->name();
  if (entriesByName().contains(name)) {
    throw std::runtime_error("Op name already registered");
  }

  entriesByType()[op] = entry;
  entriesByName()[name] = entry;
}

Ops::Entry& Ops::get(Op* op) {
  op->init();
  std::type_index idx(typeid(*op));
  auto entry = entriesByType()[idx];
  if (!entry) throw std::out_of_range("unknown Op");
  return *entry;
}

std::string Ops::name(Op* op) {
  return get(op).name();
}

std::string Ops::label(Op* op) {
  return get(op).label(op);
}

BackOps Ops::back(const BackCtx& ctx) {
  auto op = ctx.forward;
  auto& entry = get(op);
  return entry.back(ctx);
}

bool Ops::isSideEffect(Op* op) {
  return get(op).sideEffect(op);
}

Op* Ops::copy(Op* op) {
  return get(op).copy(op);
}

Ops::Entry::~Entry() {
}

}

namespace matcha::engine::ops {

std::string name(Op* op) { return Ops::name(op); }
std::string label(Op* op) { return Ops::label(op); }
BackOps back(const BackCtx& ctx) { return Ops::back(ctx); }
bool isSideEffect(Op* op) { return Ops::isSideEffect(op); }
Op* copy(Op* op) { return Ops::copy(op); }

}