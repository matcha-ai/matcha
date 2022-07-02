#pragma once

#include "bits_of_matcha/engine/cpu/Buffer.h"

#include <cstddef>
#include <initializer_list>
#include <vector>


namespace matcha::engine::cpu {

class BlockPool;

class MemoryPool {
public:
  static MemoryPool* the();

  Block* malloc(size_t bytes);
  void free(Block* block);

  size_t usage() const;

private:
  MemoryPool();

  static MemoryPool* the_;
  std::vector<BlockPool> blockPools_;

  BlockPool* bestFit(size_t bytes);

};


}