#include "bits_of_matcha/engine/pipeBuffers.h"
#include "bits_of_matcha/engine/buffer.h"
#include "bits_of_matcha/device.h"

#include <stdexcept>


namespace matcha::engine {


PipeBuffers::PipeBuffers(std::initializer_list<Buffer*> init) {
  for (auto buffer: init) push(buffer);
}

void PipeBuffers::push(Buffer* buffer) {
  if (buffer->uses(CPU)) cpuBuffers_.push(buffer);
}

Buffer* PipeBuffers::take(const Device::Concrete& device) {
  std::stack<Buffer*>* stacc;
  if (device == CPU) {
    stacc = &cpuBuffers_;
  } else {
    throw std::runtime_error("unknown device");
  }
  if (stacc->empty()) return nullptr;
  Buffer* buff = stacc->top();
  stacc->pop();
  return buff;
}


}