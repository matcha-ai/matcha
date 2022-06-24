#pragma once

#include <functional>


namespace matcha {

template <class T, template <class, class> class Map>
void apply(const Map<T*, T>& ts, const std::function<void (T&, const T&)>& handler) {
//  for (auto&& [target, data]: ts) {
//    handler(*target, data);
//  }
}


}