#pragma once

#include <cstddef>
#include <initializer_list>
#include <vector>


namespace matcha::engine::cpu {

class Buffer;
class BlockPool;

class MemoryPool {
public:
  static MemoryPool* the();

  Buffer* malloc(size_t bytes);
  void free(Buffer* buffer);

  size_t usage() const;

private:
  MemoryPool();

  static MemoryPool* the_;
  std::vector<BlockPool> blockPools_;

  BlockPool* bestFit(size_t size);

};


}