#pragma once

#include "bits_of_matcha/ops.h"

#include <set>


namespace matcha::engine {

class Module;
class Tensor;
class Op;

class Flow {
public:
  explicit Flow(const AnyOp& preimage);
  explicit Flow();

  bool hasPreimage() const;
  void setPreimage(const AnyOp& op);

  void build(const std::vector<Frame>& frames);
  std::vector<Module*> modules() const;

  std::vector<Tensor*> call(const std::vector<Tensor*>& ins);

private:
  Module* module(const std::vector<Frame>& frames);
  Module* module(const std::vector<Tensor*>& tensors);
  std::string getId(const std::vector<Frame>& frames);

private:
  AnyOp preimage_;
  std::unordered_map<std::string, Module*> modules_;
  Module* lastCalledModule_;
};

}