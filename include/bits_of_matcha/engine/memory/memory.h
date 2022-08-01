#pragma once

#include "bits_of_matcha/Device.h"
#include "bits_of_matcha/Frame.h"


namespace matcha::engine {

class Buffer;

}

namespace matcha::engine::stats {

size_t memory(const Device::Concrete& device);
size_t memory();

}