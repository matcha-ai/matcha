#pragma once

#include <string>


namespace matcha {

class Object;

namespace engine {


class Object {
  public:

    const std::string& name() const;
    void rename(const std::string& name) const;

    bool referenced() const;

  protected:
    Object();
    Object(const std::string& name);
    virtual ~Object();

  private:
    mutable unsigned refCount_;
    mutable std::string name_;

    void bindRef(const matcha::Object* ref);
    void unbindRef(const matcha::Object* ref);

    virtual void considerPruning() = 0;

    friend class matcha::Object;
};


}
}
