#pragma once

#include "bits_of_matcha/engine/op/Ops.h"
#include "bits_of_matcha/engine/op/BackCtx.h"
#include "bits_of_matcha/print.h"

#include <string>
#include <functional>


namespace matcha::engine {

class Op;
class OpBack;
class BackCtx;

template <class T>
struct OpMeta {
  std::string name;
  std::function<std::string (T*)> label = [&](auto) { return name; };
  std::function<Op* (const BackCtx&)> back = [](auto) { return nullptr; };
  bool sideEffect = false;
  std::function<void (T*)> save;
  std::function<void (T*)> load;

  struct RegisterCtx {
    explicit RegisterCtx(OpMeta* meta) {

      auto entry = new Ops::Entry {
        .name = [meta] { return meta->name; },
        .label = [meta] (Op* op) { return meta->label(dynamic_cast<T*>(op)); },
        .back = meta->back,
        .sideEffect = [meta] (Op* op) { return meta->sideEffect; },
      };

      Ops::add(typeid(T), entry);
    }
  };

  RegisterCtx registerCtx_{this};
  ~OpMeta() {

  }
};

}