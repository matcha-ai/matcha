#include "bits_of_matcha/engine/debug.h"
#include "bits_of_matcha/engine/object.h"


#include <matcha/engine>

namespace matcha {
namespace engine {


Debug::Debug()
  : context_{Context::current()}
  , level_{context_->getDebug()}
{
  if (!level_) return;
  std::cout << "<" << context_->name() << "> ";
}

Debug::~Debug() {
  if (!level_) return;
  std::cout << std::endl;
}

Debug& Debug::operator<<(const char* c) {
  if (!level_) return *this;
  std::cout << c;
  return *this;
}

Debug& Debug::operator<<(const std::string &s) {
  if (!level_) return *this;
  std::cout << s;
  return *this;
}

Debug& Debug::operator<<(float f) {
  if (!level_) return *this;
  printf("%.3f", f);
  return *this;
}

Debug& Debug::operator<<(int d) {
  if (!level_) return *this;
  printf("%d", d);
  return *this;
}

Debug& Debug::operator<<(size_t zu) {
  if (!level_) return *this;
  printf("%zu", zu);
  return *this;
}

Debug& Debug::operator<<(bool b) {
  if (!level_) return *this;
  std::cout << (b ? "True" : "False");
  return *this;
}

Debug& Debug::operator<<(const void* ptr) {
  if (!level_) return *this;
  std::cout << ptr;
  return *this;
}

Debug& Debug::operator<<(const Object* object) {
  if (!level_) return *this;
  std::cout << object->name();
  return *this;
}

Debug& Debug::operator<<(const Dtype& dtype) {
  if (!level_) return *this;
  std::cout << dtype;
  return *this;
}

Debug& Debug::operator<<(const Shape& shape) {
  if (!level_) return *this;
  std::cout << shape;
  return *this;
}

}
}
