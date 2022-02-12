#pragma once

#include <matcha/device>
#include <stack>


namespace matcha {

class Context {
  public:
    Context();
    Context(const Device& device);

    void use(const Device& device);

    ~Context();

  public:
    static const Device& device();

  private:
    const Device* device_;

  private:
    static Context* parentContext();
    void pushContext();
    void popContext();

  private:
    static thread_local std::stack<Context*> contextStack_;
    static Context defaultContext_;

  private:
    void* operator new(size_t bytes) = delete;
    void* operator new[](size_t bytes) = delete;
    void operator delete(void* ptr) = delete;
    void operator delete[](void* ptr) = delete;
};

}
