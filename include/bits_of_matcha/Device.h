#pragma once

#include <functional>
#include <variant>


namespace matcha::engine {
class Buffer;
}

namespace matcha {

class Computation;

class Device {
public:
  struct Concrete {
    Concrete(unsigned type);

    unsigned type;

    bool hosts(engine::Buffer* buffer) const;

    bool operator==(const Concrete& device) const;
    bool operator!=(const Concrete& device) const;
  };
  using Strategy = std::function<Concrete (const Computation& c)>;

  enum {
    CPU,
    GPU
  };

  Device(unsigned type);
  Device(Strategy strategy);

  Concrete get(const Computation& c) const;

private:
  std::variant<Concrete, Strategy> device_;
};

enum {
  CPU = Device::CPU,
  GPU = Device::GPU
};


}