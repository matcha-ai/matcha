#include "bits_of_matcha/error/Error.h"


namespace matcha {

Error::Error(const std::string& message)
  : std::runtime_error(message)
{}

Error::Error(const char* message)
  : std::runtime_error(message)
{}

Error::Error()
  : std::runtime_error("")
{}

}