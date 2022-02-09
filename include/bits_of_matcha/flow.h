#pragma once

#include "bits_of_matcha/object.h"

#include <initializer_list>
#include <vector>
#include <string>
#include <iostream>

namespace matcha {
  class Flow;
}

std::ostream& operator<<(std::ostream& os, const matcha::Flow& flow);

namespace matcha {

class Tuple;
class Tensor;
class Stream;

class Device;

namespace engine {
  class Flow;
}

class Flow : public Object {
  public:
    Flow(const Tensor& out);
    Flow(const Tuple& outs);
    Flow(std::initializer_list<Tensor> outs);

    Tensor operator()(const Tensor& in) const;
    Tuple operator()(const Tuple& ins) const;

    void use(const Device& device);
    void test(const Stream& stream) const;

    void save(const std::string& filepath) const;
    void save(std::ostream& os) const;

    static Flow load(const std::string& filepath);
    static Flow load(std::istream& is);

  public:
    static Flow fromObject(engine::Flow* object);

  private:
    Flow(engine::Flow* object, char dummy);
    engine::Flow* object() const;

  private:
    friend std::ostream& ::operator<<(std::ostream& os, const Flow& flow);

};

}
