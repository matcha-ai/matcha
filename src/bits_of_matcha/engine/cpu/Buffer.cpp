#include "bits_of_matcha/engine/cpu/Buffer.h"
#include "bits_of_matcha/engine/cpu/MemoryPool.h"

#include <iostream>


namespace matcha::engine::cpu {


Buffer::Buffer(size_t bytes)
  : engine::Buffer{CPU, bytes}
  , memory_{new uint8_t[bytes]}
{
}

Buffer::Buffer(size_t bytes, void* memory)
  : engine::Buffer{CPU, bytes}
  , memory_{(uint8_t *)memory}
{}

Buffer::~Buffer() {
  MemoryPool::the()->free(this);
}

void* Buffer::payload() {
  return memory_;
}


}