#pragma once

#include <string>


namespace matcha {
namespace engine {
  class Object;
}


class Object {
  public:
    const std::string& name() const;
    void rename(const std::string& name) const;

  protected:
    Object();
    Object(engine::Object* object);
    Object(const Object& other);
    virtual ~Object();

    void reset(engine::Object* object) const;
    void release() const;
    engine::Object* object() const;
    bool isNull() const;

    const Object& operator=(const Object& other);

  private:
    mutable engine::Object* object_;

    friend class engine::Object;

};


}
