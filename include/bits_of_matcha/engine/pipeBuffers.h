#pragma once

#include "bits_of_matcha/device.h"

#include <stack>


namespace matcha::engine {

class Buffer;

class PipeBuffers {
  public:
    PipeBuffers() = default;
    PipeBuffers(std::initializer_list<Buffer*> init);

    Buffer* take(const Device::Concrete& device);
    void push(Buffer* buffer);

  private:
    std::stack<Buffer*> cpuBuffers_;

};


}