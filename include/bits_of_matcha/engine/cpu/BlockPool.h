#pragma once

#include <cstddef>
#include <stack>


namespace matcha::engine::cpu {

class Block;


class BlockPool {
public:
  BlockPool(size_t blockSize, size_t quantum);
  Block* allocate();
  void free(Block* buffer);

  size_t blockSize() const;
  size_t blocks() const;

private:
  size_t block_size_;
  size_t quantum_;
  size_t blocks_;

  std::stack<void*> freeBlocks_;

  void expandBlocks();


};


}