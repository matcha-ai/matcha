#include "bits_of_matcha/print.h"
#include "bits_of_matcha/tensor.h"


namespace std {
matcha::Mout mout;
}

namespace matcha {

Mout& Mout::operator<<(const tensor& tensor) {
  if (!ss.str().empty()) {
    std::cout << ss.str();
    ss.str("");
  }
  ::operator<<(std::cout, tensor);
  return *this;
}

Mout& Mout::operator<<( std::ostream& (*f)(std::ostream&) ) {
  if (f == static_cast<std::ostream& (*)(std::ostream&)>(&std::endl)) {
    std::cout << ss.str() << std::endl;
    ss.str("");
  } else if (f == static_cast<std::ostream& (*)(std::ostream&)>(&std::flush)) {
    std::cout << ss.str() << std::flush;
    ss.str("");
  }
  return *this;
}

}