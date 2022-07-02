#include "bits_of_matcha/engine/memory/Block.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/print.h"


namespace matcha::engine {

void Block::transfer(Block* source, Block* target) {
  if (source->device()->type == CPU && target->device()->type == CPU) {
    auto a = (uint8_t*) source->payload();
    auto b = (uint8_t*) target->payload();
    std::copy(a, a + source->bytes_, b);
  } else {
//    std::cout << script->refs_ << " -> ";
//    std::cout << script->device()->type << " -> ";
//    std::cout << target->device()->type << std::endl;
    throw std::runtime_error("gpu not implemented yet");
  }
//  std::cout << "transfer done" << std::endl;
}

Block::Block(const Device::Concrete& device, size_t bytes)
: dev_ {device}, bytes_ {bytes}, refs_ {0} {}

Block::~Block() {
//  print("deleting buffer");
}

const Device::Concrete* Block::device() const {
  return &dev_;
}

size_t Block::bytes() const {
  return bytes_;
}

bool Block::fits(size_t bytes) const {
  return bytes <= bytes_;
}

bool Block::uses(const Device::Concrete& d) const {
  return d == dev_;
}

bool Block::shared() const {
  return refs_ > 1;
}

bool Block::writable() const {
  return !shared();
}

bool Block::bound() const {
  return refs_;
}

void Block::bind() {
  refs_++;
}

void Block::unbind() {
  if (!refs_) throw std::runtime_error("Block refs are already 0");
  refs_--;
  if (!refs_) delete this;
}

}