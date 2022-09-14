#pragma once

#include "bits_of_matcha/engine/transform/Transform.h"
#include "bits_of_matcha/engine/lambda/Module.h"

#include <map>


namespace matcha::engine {

class CachingTransform : public Transform {
public:
  explicit CachingTransform(const fn& function);
  explicit CachingTransform() = default;

  std::vector<Tensor*> run(const std::vector<Tensor*>& inputs) override;

  void build(const std::vector<Tensor*>& tensors);
  void build(const std::vector<Frame>& frames);

  std::shared_ptr<Executor> cache(const std::vector<Tensor*>& tensors);
  std::shared_ptr<Executor> cache(const std::vector<Frame>& frames);

protected:
  virtual std::shared_ptr<Executor> compile(Lambda lambda);

private:
  std::map<std::string, std::shared_ptr<Executor>> cache_;
  static std::string hash(const std::vector<Frame>& frames);
  static std::vector<Frame> frames(const std::vector<Tensor*>& tensors);

};

}