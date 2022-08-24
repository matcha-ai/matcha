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

Lambda Registry::back(const BackCtx& ctx) {
  auto op = ctx.forward;
  auto& entry = get(op);

  std::vector<Frame> grad_frames;
  grad_frames.reserve(ctx.vals.size());
  for (auto&& grad: ctx.vals)
    grad_frames.push_back(grad->frame());

  Tracer tracer;
  auto ins = tracer.open(grad_frames);

  auto insInternal = deref(ins);
  BackCtx innerCtx {
    .forward = ctx.forward,
    .vals = insInternal,
    .wrts = ctx.wrts,
  };
  auto outsInternal = entry.back(innerCtx);
  auto lambda1 = tracer.close(engine::ref(outsInternal));
  Lambda lambda2;

  std::set<Tensor*> tensors(lambda1.tensors.begin(), lambda1.tensors.end());
  for (auto&& lop: lambda1.ops) {
    for (auto&& in: lop->inputs) {
      if (!in) continue;
      if (tensors.contains(in)) continue;
      tensors.insert(in);
      lambda1.tensors.push_back(in);
      in->req();
    }
    for (auto&& out: lop->outputs) {
      if (!out) continue;
      if (tensors.contains(out)) continue;
      tensors.insert(out);
      lambda1.tensors.push_back(out);
      out->req();
    }
  }

  for (int i = 0; i < lambda1.inputs.size(); i++) {
    auto in1 = lambda1.inputs[i];
    auto in2 = ctx.vals[i];
    auto id = new ops::Identity(in2, in1);
    lambda2.inputs.push_back(in2);
    lambda2.tensors.push_back(in2);
    in2->req();
    lambda2.ops.push_back(id);
  }

  for (auto&& opp: lambda1.ops) lambda2.ops.push_back(opp);
  for (auto&& t: lambda1.tensors) lambda2.tensors.push_back(t);
  for (auto&& out: lambda1.outputs) lambda2.outputs.push_back(out);

  lambda1 = {};

  return lambda2;
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
Lambda back(const BackCtx& ctx) { return Registry::back(ctx); }
bool isSideEffect(Op* op) { return Registry::isSideEffect(op); }
Op* copy(Op* op) { return Registry::copy(op); }

}