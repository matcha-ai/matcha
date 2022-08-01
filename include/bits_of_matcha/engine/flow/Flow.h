#pragma once

#include "bits_of_matcha/ops.h"

#include <set>
#include <memory>


namespace matcha::engine {

class Module;
class Tensor;
class Chain;
class Op;

class Flow final {
public:
  explicit Flow(const AnyOp& preimage);
  explicit Flow();
  ~Flow() = default;

  bool hasPreimage() const;
  void setPreimage(const AnyOp& op);

  void build(const std::vector<Frame>& frames);
  std::vector<Module*> modules() const;

  std::vector<Tensor*> call(const std::vector<Tensor*>& ins);

  void ref();
  void unref();
  unsigned refs() const;

private:
  std::shared_ptr<Module> module(const std::vector<Frame>& frames);
  std::shared_ptr<Module> module(const std::vector<Tensor*>& tensors);
  std::string getId(const std::vector<Frame>& frames);

  static void optimizer(Chain& chain);

private:
  AnyOp preimage_;
  std::unordered_map<std::string, std::shared_ptr<Module>> modules_;
  std::shared_ptr<Module> lastCalledModule_;

private:
  unsigned refs_;
};

}