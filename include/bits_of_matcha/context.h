#pragma once

#include <matcha/device>
#include <stack>


namespace matcha {

class Context {
  public:
    Context();
    Context(const std::string& name);

    const std::string& name() const;
    void rename(const std::string& name);

    void use(Device& device);
    void debug(int level);

    ~Context();

  public:
    static const Context* current();
    const Device* getDevice() const;
    int getDebug() const;

  private:
    std::string name_;
    const Device* device_;
    int debug_;

  private:
    static Context* parentContext();
    void pushContext();
    void popContext();

  private:
    static thread_local std::stack<Context*> contextStack_;

    Context(Device& device, int debugLevel);
    static Context defaultContext_;

  private:
    void* operator new(size_t bytes) = delete;
    void* operator new[](size_t bytes) = delete;
    void operator delete(void* ptr) = delete;
    void operator delete[](void* ptr) = delete;
};

}
