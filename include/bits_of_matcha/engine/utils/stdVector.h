#pragma once

#include <vector>

namespace matcha::engine {


//template <class T, class... Vectors>

template <class T, class... Vectors>
inline std::vector<T> cat(Vectors... vectors);

template <class T, class... Vectors>
inline void catInternal(std::vector<T>& buff, Vectors... vectors);

template <class T, class... Vectors>
inline std::vector<T> cat(const std::vector<T>& v, Vectors... vectors) {
  auto buff = v;
  catInternal(buff, vectors...);
  return buff;
}

template <class T, class... Vectors>
inline void catInternal(std::vector<T>& buff, const std::vector<T>& v, Vectors... vs) {
  std::copy(v.begin(), v.end(), std::back_inserter(buff));
  catInternal(buff, vs...);
}

template <class T, class... Vectors>
inline void catInternal(std::vector<T>& buff) {
}


}