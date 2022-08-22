#include "bits_of_matcha/engine/cpu/Block.h"
#include "bits_of_matcha/engine/cpu/MemoryPool.h"
#include "bits_of_matcha/print.h"


namespace matcha::engine::cpu {


Block::Block(size_t bytes)
  : engine::Block{CPU, bytes}
  , memory_{new uint8_t[bytes]}
{
}

Block::Block(size_t bytes, void* memory)
  : engine::Block{CPU, bytes}
  , memory_{(uint8_t *)memory}
{}

Block::~Block() {
  MemoryPool::the()->free(this);
}

void* Block::payload() {
  return memory_;
}


}