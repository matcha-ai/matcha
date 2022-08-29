#include "bits_of_matcha/engine/lambda/Tracer.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/ops/Print.h"
#include "bits_of_matcha/engine/ops/SideOutput.h"


namespace matcha::engine {

thread_local  std::stack<Tracer*> Tracer::tracings_ = {};
std::map<tensor*, Tensor*> Tracer::restore_state_ = {};

bool tracing() {
  return Tracer::active();
}

void incept(Op* op, Op* preop) {
  auto in = preop->inputs[0];

  auto it = std::find(op->inputs.begin(), op->inputs.end(), in);
  *it = preop->outputs[0];
  in->unreq();
  preop->outputs[0]->req();

  dispatch(preop);
  if (tracing()) {
    Tracer::handleNewTensor(preop->outputs[0]);
  }
}

Lambda trace(const fn& function, const std::vector<Frame>& frames) {
  Tracer tracer;
  tuple ins = tracer.open(frames);
  tuple outs = function(ins);
  return tracer.close(outs);
}

bool Tracer::active() {
  if (tracings_.empty()) return false;
  return !tracings_.top()->frozen_;
}

Tracer* Tracer::get() {
  if (!active()) return nullptr;
  return tracings_.top();
}

Tracer::Tracer()
  : frozen_(false)
{}

Tracer::~Tracer() {}

void Tracer::setFrozen(bool frozen) {
  frozen_ = frozen;
}

tuple Tracer::open(const std::vector<Frame>& frames) {
//  ops::Print::claimCout();
  tracings_.push(this);
  tuple inputs;
  inputs.reserve(frames.size());
  for (auto& frame: frames) {
    auto in = new Tensor(frame);
    lambda_.inputs.push_back(in);
    inputs.push_back(ref(in));
  }
  refs_.clear();
  derefs_.clear();
  return inputs;
}

Lambda Tracer::close(const tuple& outputs) {
//  ops::Print::unclaimCout();

  if (tracings_.top() != this)
    throw std::runtime_error("tracing stack got corrupted");
  tracings_.pop();

  for (auto& output: outputs) {
    auto out = deref(output);
    lambda_.outputs.push_back(out);
    auto&& temp = const_cast<tensor*>(&output);
    refs_.erase(temp);
    restore_state_.erase(temp);
  }


  // populate side outputs
  for (auto&& ref: refs_) {
    auto so = new ops::SideOutput(deref(ref), ref);
    lambda_.ops.push_back(so);
  }

  // populate side inputs
  for (auto&& [external, internal]: derefs_) {
    if (side_inputs_.contains(internal))
      lambda_.side_inputs[internal] = external;
  }

  // restore state if no longer tracing anything

  if (tracings_.empty()) {
    for (auto&& [lhs, state]: restore_state_) {
      *lhs = ref(state);
      if (state) state->unreq();
    }
    restore_state_.clear();
  }

  return std::move(lambda_);
}

bool Tracer::handleNewOp(Op* op) {
  auto tracer = get();
  if (!tracer) return false;
  return tracer->addNewOp(op);
}

bool Tracer::handleNewTensor(Tensor* tensor) {
  auto tracer = get();
  if (!tracer) return false;
  return tracer->addNewTensor(tensor);
}

bool Tracer::handleOldTensor(Tensor* tensor) {
  auto tracer = get();
  if (!tracer) return false;
  return tracer->addOldTensor(tensor);
}

bool Tracer::handleNewRef(tensor* ref, Tensor* internal) {
  auto tracer = get();
  if (!tracer) return false;
  return tracer->addNewRef(ref, internal);
}

bool Tracer::handleDelRef(tensor* ref, Tensor* internal) {
  auto tracer = get();
  if (!tracer) return false;
  return tracer->addDelRef(ref, internal);
}

bool Tracer::handleNewDeref(const tensor* external, Tensor* internal) {
  auto tracer = get();
  if (!tracer) return false;
  return tracer->addNewDeref(external, internal);
}

bool Tracer::addNewOp(Op* op) {
  lambda_.ops.push_back(op);
  return true;
}

bool Tracer::addNewTensor(Tensor* tensor) {
  if (added_tensors_.contains(tensor)) return true;
  added_tensors_.insert(tensor);
  lambda_.tensors.push_back(tensor);
  tensor->req();
  return true;
}

bool Tracer::addOldTensor(Tensor* tensor) {
  if (added_tensors_.contains(tensor)) return true;
  added_tensors_.insert(tensor);
  side_inputs_.insert(tensor);
  lambda_.tensors.push_back(tensor);
  tensor->req();
  return true;
}

bool Tracer::addNewRef(tensor* ref, Tensor* internal) {
  if (!internal) return true;
  if (!refs_.contains(ref))
    refs_.insert(ref);
  if (!restore_state_.contains(ref)) {
    restore_state_[ref] = internal;
    if (internal) internal->req();
  }
  return true;
}

bool Tracer::addDelRef(tensor* ref, Tensor* internal) {
  if (refs_.contains(ref))
    refs_.erase(ref);

  if (derefs_.contains(ref))
    derefs_.erase(ref);

  if (restore_state_.contains(ref))
    restore_state_.erase(ref);

  if (side_inputs_.contains(internal))
    side_inputs_.erase(internal);

  return true;
}

bool Tracer::addNewDeref(const tensor* external, Tensor* internal) {
  if (!derefs_.contains(external))
    derefs_[external] = internal;
//  if (!derefs_.contains(internal))
//    derefs_[internal] = external;
  return true;
}

}