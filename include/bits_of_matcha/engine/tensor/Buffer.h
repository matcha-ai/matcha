#pragma once

#include "bits_of_matcha/Frame.h"
#include "bits_of_matcha/engine/memory/Block.h"


namespace matcha::engine {

class Buffer {
public:
  explicit Buffer();
  explicit Buffer(size_t bytes);
  explicit Buffer(Block* block);

  Buffer(const Buffer& other);
  Buffer(Buffer&& other) noexcept;
  ~Buffer();

  Buffer& operator=(const Buffer& other);
  Buffer& operator=(Buffer&& other) noexcept;

  bool operator==(const Buffer& other);
  bool operator!=(const Buffer& other);

  operator bool() const;
  size_t bytes() const;

  void malloc(size_t bytes);
  void free();

  void* payload();

  template <class T>
  inline T as() { return reinterpret_cast<T>(payload()); }

private:
  Block* block_;
};

}