#pragma once

#include <functional>


namespace matcha::engine {
using Instruction = std::function<void ()>;
}