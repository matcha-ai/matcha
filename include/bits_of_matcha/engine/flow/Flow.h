#pragma once

#include "bits_of_matcha/engine/Tensor.h"
#include "bits_of_matcha/engine/flow/Instructions.h"


namespace matcha::engine {

class Instructions;

class Flow {
public:
  std::vector<Tensor*> run(const std::vector<Tensor*> ins);

  std::vector<Frame> signatureIns() const;
  std::vector<Frame> signatureOuts() const;

private:
  Instructions instructions_;
};

}