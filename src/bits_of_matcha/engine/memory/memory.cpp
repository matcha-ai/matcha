#include "bits_of_matcha/engine/memory/memory.h"
#include "bits_of_matcha/engine/memory/Block.h"
#include "bits_of_matcha/engine/cpu/MemoryPool.h"
#include "bits_of_matcha/engine/cpu/Block.h"

#include <stdexcept>

namespace matcha::engine::stats {

size_t memory(const Device::Concrete& device) {
  if (device.type == CPU) {
    return cpu::MemoryPool::the()->usage();
  } else {
    throw std::runtime_error("TODO: gpu memory usage");
  }
}

size_t memory() {
  size_t total = 0;
  total += memory(CPU);
  return total;
}

}