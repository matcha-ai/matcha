#include "bits_of_matcha/engine/tensor/Buffer.h"
#include "bits_of_matcha/engine/memory/memory.h"
#include "bits_of_matcha/engine/cpu/MemoryPool.h"


namespace matcha::engine {

Buffer::Buffer()
  : block_(nullptr)
{}

Buffer::Buffer(size_t bytes)
  : block_(nullptr)
{
  malloc(bytes);
}

void Buffer::malloc(size_t bytes) {
  if (block_) {
    if (block_->fits(bytes)) return;
    block_->unbind();
  }
  block_ = cpu::MemoryPool::the()->malloc(bytes);
  block_->bind();
}

void Buffer::free() {
  if (block_) {
    block_->unbind();
    block_ = nullptr;
  }
}

Buffer::Buffer(const Buffer& other) {
  block_ = other.block_;
  if (block_) block_->bind();
}

Buffer::Buffer(Buffer&& other) noexcept {
  block_ = other.block_;
  other.block_ = nullptr;
}

Buffer::Buffer(Block* block) {
  block_ = block;
  if (block_) block_->bind();
}

Buffer::~Buffer() {
  if (block_) block_->unbind();
}

Buffer& Buffer::operator=(const Buffer& other) {
  if (block_ == other.block_) return *this;
  if (block_) block_->unbind();
  block_ = other.block_;
  if (block_) block_->bind();
  return *this;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
  if (block_ == other.block_) return *this;
  if (block_) block_->unbind();
  block_ = other.block_;
  other.block_ = nullptr;
  return *this;
}

Buffer::operator bool() const {
  return block_;
}

size_t Buffer::bytes() const {
  if (!block_) return 0;
  return block_->bytes();
}

void* Buffer::payload() {
  if (!block_) return nullptr;
  return block_->payload();
}

bool Buffer::operator==(const Buffer& other) {
  return block_ == other.block_;
}

bool Buffer::operator!=(const Buffer& other) {
  return  block_ != other.block_;
}

}