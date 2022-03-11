#pragma once

#include <string>
#include <stack>


namespace matcha {

#define SCOPE_MAKE(lvalue, name) matcha::Scope lvalue(name);
#define SCOPE0() SCOPE_MAKE(__anonymous_scope__, "")
#define SCOPE1(name) SCOPE_MAKE(name##_scope, #name)
#define GET_SCOPE_MACRO(_0, _1, NAME, ...) NAME
#define scope(...) GET_SCOPE_MACRO(_0, ##__VA_ARGS__, SCOPE1, SCOPE0)(__VA_ARGS__)


class Device;
void use(const Device& device);

class Scope {
  public:
    Scope();
    Scope(const std::string& name);
    ~Scope();

    void* operator new(size_t bytes) = delete;
    void* operator new[](size_t bytes) = delete;
    void operator delete(void* ptr) = delete;
    void operator delete[](void* ptr) = delete;

  private:
    std::stack<Scope*> scopes_;
};


}