#include "bits_of_matcha/Device.h"
#include "bits_of_matcha/engine/memory/Buffer.h"


namespace matcha {


Device::Device(unsigned type)
  : device_{Concrete{type}}
{}

Device::Device(Strategy strategy)
  : device_{std::move(strategy)}
{}

Device::Concrete Device::get(const Computation& c) const {
  if (std::holds_alternative<Concrete>(device_)) {
    return std::get<Concrete>(device_);
  } else {
    return std::get<Strategy>(device_)(c);
  }
}

Device::Concrete::Concrete(unsigned type)
  : type{type}
{}

bool Device::Concrete::operator==(const Concrete& device) const {
  return device.type == type;
}

bool Device::Concrete::operator!=(const Concrete& device) const {
  return !operator==(device);
}


bool Device::Concrete::hosts(engine::Buffer* buffer) const {
  if (!buffer) return false;
  return buffer->uses(*this);
}

}