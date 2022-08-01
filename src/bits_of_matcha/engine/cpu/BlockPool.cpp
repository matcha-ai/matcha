#include "bits_of_matcha/engine/cpu/BlockPool.h"
#include "bits_of_matcha/engine/cpu/Buffer.h"
#include "bits_of_matcha/print.h"


namespace matcha::engine::cpu {

BlockPool::BlockPool(size_t blockSize, size_t quantum)
  : block_size_{blockSize}
  , quantum_{quantum}
  , blocks_{0}
{}

Block* BlockPool::allocate() {
  if (freeBlocks_.empty()) {
    expandBlocks();
  }
  auto freeBlock = freeBlocks_.top();
  freeBlocks_.pop();
  return new Block(block_size_, freeBlock);
}

void BlockPool::free(Block* buffer) {
  freeBlocks_.push(buffer->payload());
}

void BlockPool::expandBlocks() {
  auto chunk = new uint8_t[block_size_ * quantum_];
  for (size_t i = 0; i < quantum_; i++) {
    freeBlocks_.push(chunk);
    chunk += block_size_;
  }
  blocks_ += quantum_;
}

size_t BlockPool::blockSize() const {
  return block_size_;
}

size_t BlockPool::blocks() const {
  return blocks_;
}

}