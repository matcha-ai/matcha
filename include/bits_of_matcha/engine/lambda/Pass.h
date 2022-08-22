#pragma once

#include <functional>

namespace matcha::engine {

class Lambda;

using Pass = std::function<void (Lambda&)>;

}