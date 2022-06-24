#pragma once

#include "bits_of_matcha/savers/FlowPlot.h"
#include "bits_of_matcha/savers/TensorImage.h"

#include <variant>

namespace matcha {

class FlowPlot;
class TensorImage;

using SaveSpec = std::variant<
  FlowPlot,
  TensorImage
>;

}