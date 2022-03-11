#pragma once

#include <cstddef>
#include <stack>


namespace matcha::engine::cpu {

class Buffer;

class BlockPool {
  public:
    BlockPool(size_t blockSize, size_t quantum);
    Buffer* allocate();
    void free(Buffer* buffer);

    size_t blockSize() const;
    size_t blocks() const;

  private:
    size_t blockSize_;
    size_t quantum_;
    size_t blocks_;

    std::stack<void*> freeBlocks_;

    void expandBlocks();


};


}