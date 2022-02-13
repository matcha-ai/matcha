#include "bits_of_matcha/data.h"

#include <stdexcept>


namespace matcha {


Data::Data(void* data)
  : data_{data}
{}

void* Data::vs() {
  return data_;
}

float* Data::f32s() {
  return (float*) data_;
}

float* Data::fs() {
  return (float*) data_;
}

float Data::f32() {
  if (data_ == nullptr) throw std::runtime_error("data not ready yet");
  return *f32s();
}

float Data::f() {
  if (data_ == nullptr) throw std::runtime_error("data not ready yet");
  return *fs();
}

int Data::i32() {
  if (data_ == nullptr) throw std::runtime_error("data not ready yet");
  return (int) *f32s();
}

int Data::i() {
  if (data_ == nullptr) throw std::runtime_error("data not ready yet");
  return (int) *fs();
}

bool Data::b() {
  if (data_ == nullptr) throw std::runtime_error("data not ready yet");
  return (bool) *fs();
}

Data::operator float *() {
  return (float*) data_;
}

Data::operator float() {
  if (data_ == nullptr) throw std::runtime_error("data not ready yet");
  return *(float*) data_;
}

Data::operator int() {
  if (data_ == nullptr) throw std::runtime_error("data not ready yet");
  return (int)*(float*) data_;
}

Data::operator bool() {
  if (data_ == nullptr) throw std::runtime_error("data not ready yet");
  return (bool)*(float*) data_;
}

float Data::operator*() {
  return f();
}


}
