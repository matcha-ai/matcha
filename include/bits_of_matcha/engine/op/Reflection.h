#pragma once

#include "bits_of_matcha/engine/op/Registry.h"
#include "bits_of_matcha/engine/op/BackCtx.h"
#include "bits_of_matcha/engine/lambda/Lambda.h"
#include "bits_of_matcha/print.h"

#include <string>
#include <functional>


namespace matcha::engine {

class Op;
class OpBack;
class BackCtx;

template <class T>
struct Reflection {
  std::string name;
  std::function<std::string (T*)> label = [&](auto) { return name; };
  std::function<std::vector<Tensor*> (const BackCtx&)> back = [](auto&) { return std::vector<Tensor*>{}; };
  bool side_effect = false;
  std::function<void (T*)> save;
  std::function<void (T*)> load;
  std::function<T* (T*)> copy = [](T* op) { return new T(*op); };

  struct RegisterCtx {
    explicit RegisterCtx(Reflection* meta) {

      auto entry = new Registry::Entry {
        .name = [meta] { return meta->name; },
        .label = [meta] (Op* op) { return meta->label(dynamic_cast<T*>(op)); },
        .back = meta->back,
        .side_effect = [meta] (Op* op) { return meta->side_effect; },
        .copy = [meta] (Op* op) { return meta->copy(dynamic_cast<T*>(op)); },
      };

      Registry::add(typeid(T), entry);
    }
  };

  RegisterCtx register_ctx_{this};
  ~Reflection() {

  }
};

}