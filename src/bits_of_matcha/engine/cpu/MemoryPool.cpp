#include "bits_of_matcha/engine/cpu/MemoryPool.h"
#include "bits_of_matcha/engine/cpu/BlockPool.h"
#include "bits_of_matcha/engine/cpu/Buffer.h"
#include "bits_of_matcha/print.h"

#include <algorithm>


namespace matcha::engine::cpu {


MemoryPool* MemoryPool::the_ = nullptr;

MemoryPool* MemoryPool::the() {
  if (!the_) the_ = new MemoryPool();
  return the_;
}

MemoryPool::MemoryPool()
/*
      0x20,           // 32 B            8 floats
      0x200,          // 512 B         128 floats
      0x800,          // 2 kiB         512 floats
      0x2000,         // 8 kiB          2k floats
      0x8000,         // 32 kiB         8k floats
      0x20000,        // 128 kiB       32k floats
      0x80000,        // 512 kiB      128k floats
      0x200000,       // 2 MiB        512k floats
      0x800000,       // 8 MiB          2M floats
      0x2000000,      // 32 MiB         8M floats
      0x8000000,      // 128 MiB       32M floats
*/
{
  blockPools_ = {
    {0x20, 128},
    {0x200, 64},
    {0x800, 32},
    {0x2000, 16},
    {0x8000, 16},
    {0x20000, 16},
    {0x80000, 8},
    {0x200000, 8},
    {0x800000, 4},
    {0x2000000, 1},
    {0x8000000, 1},
  };
}


Buffer* MemoryPool::malloc(size_t bytes) {
  auto pool = bestFit(bytes);

  if (pool) {
    return pool->allocate();
  } else {
    return new Buffer(bytes);
  }
}

void MemoryPool::free(Buffer* buffer) {
  auto pool = bestFit(buffer->bytes());

  if (pool) {
    pool->free(buffer);
  } else {
    delete (uint8_t*) buffer->payload();
  }
}

BlockPool* MemoryPool::bestFit(size_t size) {
  auto it = std::lower_bound(
    std::begin(blockPools_), std::end(blockPools_),
    size,
    [](auto& pool, size_t required) {
      return pool.blockSize() < required;
    }
  );
  if (it == std::end(blockPools_)) return nullptr;
  return &*it;
}

size_t MemoryPool::usage() const {
  size_t total = 0;
  for (auto pool: blockPools_) {
    total += pool.blockSize() * pool.blocks();
  }
  return total;
}

}