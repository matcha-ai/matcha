#include "bits_of_matcha/error/IncompatibleDtypesError.h"

#include <sstream>


namespace matcha {

IncompatibleDtypesError::IncompatibleDtypesError(const Dtype& a, const Dtype& b)
: Error("Dtypes " + a.string() + " and " + b.string() + " are not compatible.")
{}


}
