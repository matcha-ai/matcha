#include "bits_of_matcha/engine/flow.h"


namespace matcha::engine {

Flow::Flow()
  : built_{false}
{}

bool Flow::built() {
  return built_;
}

void Flow::check() {
  built_ = true;
}

}