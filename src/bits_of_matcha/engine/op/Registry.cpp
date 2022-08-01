#include "bits_of_matcha/engine/op/Registry.h"
#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/ops/Identity.h"
#include "bits_of_matcha/print.h"


namespace matcha::engine {

std::unordered_map<std::type_index, Registry::Entry*>& Registry::entriesByType() {
  static std::unordered_map<std::type_index, Entry*> entriesByType_ = {};
  return entriesByType_;
}

std::unordered_map<std::string, Registry::Entry*>& Registry::entriesByName() {
  static std::unordered_map<std::string, Entry*> entriesByName_;
  return entriesByName_;
}

void Registry::add(std::type_index op, Entry* entry) {
//  print("Registered Op: ", entry->name(), " with hash ", op.hash_code());
//  print(entriesByName().size());
  std::string name = entry->name();
  if (entriesByName().contains(name)) {
    throw std::runtime_error("Op name already registered");
  }

  entriesByType()[op] = entry;
  entriesByName()[name] = entry;
}

Registry::Entry& Registry::get(Op* op) {
  op->init();
  std::type_index idx(typeid(*op));
  auto entry = entriesByType()[idx];
  if (!entry) throw std::out_of_range("unknown Op");
  return *entry;
}

std::string Registry::name(Op* op) {
  return get(op).name();
}

std::string Registry::label(Op* op) {
  return get(op).label(op);
}

Chain Registry::back(const BackCtx& ctx) {
  auto op = ctx.forward;
  auto& entry = get(op);

  std::vector<Frame> grad_frames;
  grad_frames.reserve(ctx.vals.size());
  for (auto&& grad: ctx.vals)
    grad_frames.push_back(grad->frame());

  Tracer tracer;
  auto ins = tracer.open(grad_frames);

  // TODO: nullptr checking
  auto insInternal = deref(ins);
  BackCtx innerCtx {
    .forward = ctx.forward,
    .vals = insInternal,
    .wrts = ctx.wrts,
  };
  auto outsInternal = entry.back(innerCtx);
  auto chain1 = tracer.close(engine::ref(outsInternal));
  Chain chain2;

  for (int i = 0; i < chain1.inputs.size(); i++) {
    auto in1 = chain1.inputs[i];
    auto in2 = ctx.vals[i];
    auto id = new ops::Identity(in2, in1);
    chain2.inputs.push_back(in2);
    chain2.tensors.push_back(in2);
    chain2.ops.push_back(id);
  }

  for (auto&& opp: chain1.ops) chain2.ops.push_back(opp);
  for (auto&& t: chain1.tensors) chain2.tensors.push_back(t);
  for (auto&& out: chain1.outputs) chain2.outputs.push_back(out);

  chain1 = {};

  return chain2;
}

bool Registry::isSideEffect(Op* op) {
  return get(op).side_effect(op);
}

Op* Registry::copy(Op* op) {
  return get(op).copy(op);
}

Registry::Entry::~Entry() {
}

}

namespace matcha::engine::ops {

std::string name(Op* op) { return Registry::name(op); }
std::string label(Op* op) { return Registry::label(op); }
Chain back(const BackCtx& ctx) { return Registry::back(ctx); }
bool isSideEffect(Op* op) { return Registry::isSideEffect(op); }
Op* copy(Op* op) { return Registry::copy(op); }

}