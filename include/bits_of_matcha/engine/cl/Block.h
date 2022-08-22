#pragma once

#include "bits_of_matcha/engine/memory/Block.h"
#include "clblast.h"

namespace matcha::engine::cl {

class Block : engine::Block {
public:
  explicit Block(size_t bytes);
  explicit Block(size_t bytes, void* memory);
  ~Block() override;

  void* payload() override;

private:
  cl_mem memory;
};

}