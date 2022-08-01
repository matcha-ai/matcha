#pragma once

#include "bits_of_matcha/fn.h"
#include "bits_of_matcha/engine/chain/Chain.h"

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

class Tracer final {
  friend Chain trace(const fn&, const std::vector<Frame>&);
  friend bool tracing();
  friend void incept(Op*, Op*);

  thread_local static std::stack<Tracer*> tracings_;
  static bool active();
  static Tracer* get();

  Chain chain_;
  std::set<Tensor*> addedTensors_;

public:
  Tracer();
  ~Tracer();

  tuple open(const std::vector<Frame>& frames);
  Chain close(const tuple& outputs);

private:
  bool addNewOp(Op* op);
  bool addNewTensor(Tensor* tensor);
  bool addOldTensor(Tensor* tensor);

public:
  static bool handleNewOp(Op* op);
  static bool handleNewTensor(Tensor* tensor);
  static bool handleOldTensor(Tensor* tensor);
};

}