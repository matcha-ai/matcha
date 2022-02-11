#pragma once

#include "bits_of_matcha/engine/status.h"

#include <string>


namespace matcha {

class Object;

class Dtype;
class Shape;

namespace engine {

class In;
class Out;
class Status;

class Object {
  public:

    const std::string& name() const;
    void rename(const std::string& name) const;

  public:
    virtual void dataStatusChanged(In* in);
    virtual void updateStatusChanged(In* in);
    virtual void bufferChanged(In* in);
    virtual void eval(Out* out);
    virtual void prune(Out* out = nullptr) = 0;

    const Status& status() const;

  protected:
    Object();
    Object(const std::string& name);
    virtual ~Object();

    In* createIn(Out* source, unsigned id = 0);
    Out* createOut(const Dtype& dtype, const Shape& shape, unsigned id = 0);
    bool referenced() const;

    Status status_;

    static void release(const matcha::Object* object);
    static void release(const matcha::Object& object);

  private:
    mutable unsigned refCount_;
    mutable std::string name_;

    void bindRef(const matcha::Object* ref);
    void unbindRef(const matcha::Object* ref, bool prune = true);

    friend class matcha::Object;
};


}
}
