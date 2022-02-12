#pragma once

#include "bits_of_matcha/dtype.h"
#include "bits_of_matcha/shape.h"

#include <iostream>
#include <string>
#include <cstddef>


namespace matcha {

class Context;

namespace engine {

class Object;

class Debug {
  public:
    Debug();

    Debug& operator<<(const char* c);
    Debug& operator<<(const std::string& s);
    Debug& operator<<(float f);
    Debug& operator<<(int d);
    Debug& operator<<(size_t zu);
    Debug& operator<<(bool b);
    Debug& operator<<(const void *ptr);
    Debug& operator<<(const Object* object);
    Debug& operator<<(const Dtype& dtype);
    Debug& operator<<(const Shape& shape);

    ~Debug();

  private:
    const Context* context_;
    int level_;
};

}
}
