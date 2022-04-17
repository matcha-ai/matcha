#pragma once

#include "bits_of_matcha/Device.h"
#include <cinttypes>
#include <limits>


namespace matcha::engine {

class Tensor;


class Buffer {
public:
  static Buffer* request(const Device::Concrete& device, size_t bytes);
  static void transfer(Buffer* a, Buffer* b);

  Buffer(const Device::Concrete& device, size_t bytes);

  virtual ~Buffer();
  virtual void* payload() = 0;

  const Device::Concrete* device() const;
  size_t bytes() const;

  template <class T>
  inline T as() { return (T) payload(); }

  bool fits(size_t bytes) const;
  bool uses(const Device::Concrete& device) const;
  bool writable() const;
  bool shared() const;
  bool bound() const;

public:
  void bind();
  void unbind();

protected:
  Device::Concrete dev_;
  size_t bytes_;

private:
  uint16_t refs_;
};


}