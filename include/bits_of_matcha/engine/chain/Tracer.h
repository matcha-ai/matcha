#pragma once

#include "bits_of_matcha/fn.h"
#include "bits_of_matcha/engine/chain/Chain.h"
#include "bits_of_matcha/engine/tensor/Binding.h"

#include <stack>
#include <vector>
#include <set>
#include <map>
#include <memory>

namespace matcha::engine {

class Chain;

Chain trace(const fn& function, const std::vector<Frame>& frames);
bool tracing();

void incept(Op* op, Op* preop);

class Tracer {
  friend Chain trace(const fn&, const std::vector<Frame>&);
  friend bool tracing();
  friend void incept(Op*, Op*);

  thread_local static std::stack<Tracer*> tracings_;
  static bool active();
  static Tracer* get();

  Chain chain_;
  std::set<Tensor*> added_tensors_;
  std::set<Tensor*> side_inputs_;
  std::set<tensor*> refs_;
  std::map<const tensor*, Tensor*> derefs_;
  static std::map<tensor*, engine::Tensor*> restore_state_;
  bool frozen_;

public:
  Tracer();
  ~Tracer();

  void setFrozen(bool frozen);
  tuple open(const std::vector<Frame>& frames);
  Chain close(const tuple& outputs);

private:
  bool addNewOp(Op* op);
  bool addNewTensor(Tensor* tensor);
  bool addOldTensor(Tensor* tensor);
  bool addNewRef(tensor* binding, Tensor* internal);
  bool addDelRef(tensor* binding, Tensor* internal);
  bool addNewDeref(const tensor* external, Tensor* internal);

public:
  static bool handleNewOp(Op* op);
  static bool handleNewTensor(Tensor* tensor);
  static bool handleOldTensor(Tensor* tensor);

  static bool handleNewRef(tensor* binding, Tensor* internal);
  static bool handleDelRef(tensor* binding, Tensor* internal);
  static bool handleNewDeref(const tensor* external, Tensor* internal);
};

}