#include "bits_of_matcha/engine/buffer.h"
#include "bits_of_matcha/engine/tensor.h"
#include "bits_of_matcha/engine/cpu/buffer.h"
#include "bits_of_matcha/print.h"



namespace matcha::engine {

void Buffer::transfer(Buffer* source, Buffer* target) {
  if (source->device()->type == CPU && target->device()->type == CPU) {
    auto a = (uint8_t*) source->payload();
    auto b = (uint8_t*) target->payload();
    std::copy(a, a + source->bytes_, b);
  } else {
//    std::cout << source->refs_ << " -> ";
//    std::cout << source->device()->type << " -> ";
//    std::cout << target->device()->type << std::endl;
    throw std::runtime_error("gpu not implemented yet");
  }
//  std::cout << "transfer done" << std::endl;
}

Buffer::Buffer(const Device::Concrete& device, size_t bytes)
  : dev_{device}
  , bytes_{bytes}
  , refs_{0}
{}

Buffer::~Buffer() {
}

const Device::Concrete* Buffer::device() const {
  return &dev_;
}

size_t Buffer::bytes() const {
  return bytes_;
}

bool Buffer::fits(size_t bytes) const {
  return bytes <= bytes_;
}

bool Buffer::uses(const Device::Concrete& d) const {
  return d == dev_;
}

bool Buffer::shared() const {
  return refs_ > 1;
}

bool Buffer::writable() const {
  return !shared();
}

bool Buffer::bound() const {
  return refs_;
}

void Buffer::bind() {
  refs_++;
}

void Buffer::unbind() {
  refs_--;
  if (!refs_) delete this;
}

}