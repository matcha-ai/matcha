#pragma once

#include "bits_of_matcha/object.h"

#include <ostream>
#include <vector>


namespace matcha {
  class Tensor;
  class Stream;
}

matcha::Stream& operator>>(matcha::Stream& stream, matcha::Tensor& tensor);

namespace matcha {

class Dtype;
class Shape;

class Input;
class Tuple;
class Stream;
class Params;

class Device;

namespace engine {
  class Tensor;
  class Flow;
  class Node;
  class FlowSaver;
}

class Tensor : public Object {
  public:
    Tensor(const Dtype& dtype, const Shape& shape);
    Tensor(const Input& input);
    Tensor(const Stream& stream);
    Tensor(const Params& params);

    Tensor(float scalar);

    Tensor(const std::vector<float>& content);
    Tensor(const std::vector<std::vector<float>>& content);
    Tensor(const std::vector<std::vector<std::vector<float>>>& content);
    Tensor(const std::vector<std::vector<std::vector<std::vector<float>>>>& content);

    Tensor(std::initializer_list<float> content);
    Tensor(std::initializer_list<std::vector<float>> content);
    Tensor(std::initializer_list<std::vector<std::vector<float>>> content);
    Tensor(std::initializer_list<std::vector<std::vector<std::vector<float>>>> content);

    const Dtype& dtype() const;
    const Shape& shape() const;

    size_t rank() const;
    size_t size() const;

    void use(const Device& device) const;
    void update() const;

    void subst(const Tensor& source);
    void subst();

  public:
    static Tensor fromObject(engine::Tensor* object);

  private:
    Tensor(engine::Tensor* object, char dummy);
    engine::Tensor* object() const;

    friend Stream& ::operator>>(matcha::Stream& stream, matcha::Tensor& tensor);
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

    friend class Params;
    friend class Stream;
    friend class Input;
    friend class Stream;
    friend class engine::Flow;
    friend class engine::FlowSaver;
    friend class engine::Tensor;
    friend class engine::Node;

};


std::ostream& operator<<(std::ostream& os, const Tensor& tensor);


}
