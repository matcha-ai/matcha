#include "bits_of_matcha/error/BroadcastError.h"


namespace matcha {

BroadcastError::BroadcastError(const Shape& a, const Shape& b, int dim)
  : IncompatibleShapesError(a, b, {a.rank() + dim, b.rank() + dim})
{}

}