#pragma once

#include <iostream>
#include <sstream>

namespace matcha {

class Tensor;

class Mout {
  public:
    template <class T>
    inline Mout& operator<<(const T& t) {
      ss << t;
      return *this;
    }

    Mout& operator<<(const Tensor& tensor);
    Mout& operator<<( std::ostream& (*f)(std::ostream&) );

  private:
    std::stringstream ss;
};

}

namespace std {
extern matcha::Mout mout;
}


template <class... Args>
inline void print(Args... args);



template <class Type>
inline void PPTYPE(const Type& type) {
  std::cout << type;
}

template <>
inline void PPTYPE(const bool& type) {
  std::cout << (type ? "true" : "false");
}

template <class Arg, class... Args>
inline void print(const Arg& arg, Args... args) {
  PPTYPE(arg);
  print(args...);
}

template <>
inline void print() {
  std::cout << std::endl;
}