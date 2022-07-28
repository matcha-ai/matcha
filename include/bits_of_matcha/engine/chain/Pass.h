#pragma once

#include <functional>

namespace matcha::engine {

class Chain;

using Pass = std::function<void (Chain&)>;

}