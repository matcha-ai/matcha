#pragma once

#include "bits_of_matcha/device.h"
#include "bits_of_matcha/frame.h"
#include "bits_of_matcha/engine/tensor.h"


namespace matcha::engine {

class Buffer;

Buffer* malloc(size_t bytes, const Device::Concrete& device = CPU);
Buffer* malloc(const Frame& frame, const Device::Concrete& device = CPU);
Buffer* malloc(const Frame* frame, const Device::Concrete& device = CPU);

}

namespace matcha::engine::stats {

size_t memory(const Device::Concrete& device);
size_t memory();

}