#pragma once

#include "bits_of_matcha/engine/memory/Buffer.h"


namespace matcha::engine::cpu {

class Buffer : public engine::Buffer {
  public:
    explicit Buffer(size_t bytes);
    explicit Buffer(size_t bytes, void* memory);
    ~Buffer() override;

    void* payload() override;

  private:
    uint8_t* memory_;
};


}