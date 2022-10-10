#pragma once

#include "bits_of_matcha/ops.h"
#include "bits_of_matcha/fn.h"
#include "bits_of_matcha/engine/op/Op.h"

#include <map>
#include <vector>


namespace matcha::engine {

class Module;
class Tensor;

class Transform {
public:
  explicit Transform(fn  preimage);
  explicit Transform();

  auto preimage() -> fn&;
  auto preimage() const -> const fn&;

  void setPreimage(const fn& preimage);
  bool hasPreimage() const;

  virtual std::vector<Tensor*> run(const std::vector<Tensor*>& inputs);

private:
  fn preimage_;
};

fn ref(std::shared_ptr<Transform> transform);

}