#pragma once

#include "bits_of_matcha/object.h"

#include <ostream>
#include <vector>
#include <initializer_list>


namespace matcha {
  class Input;
  class Stream;
}

std::ostream& operator<<(std::ostream& os, const matcha::Input& input);
matcha::Stream& operator>>(matcha::Stream& stream, matcha::Input& input);

namespace matcha {

class Dtype;
class Shape;

class Stream;

namespace engine {
  class Input;
  class Tensor;
}

class Input : public Object {
  public:
    Input();
    Input(const Dtype& dtype, const Shape& shape);
    Input(const Dtype& dtype, const Shape& shape, const std::byte* content);
    Input(const Stream& stream);

    Input(float scalar);

    Input(const std::vector<float>& content);
    Input(const std::vector<std::vector<float>>& content);
    Input(const std::vector<std::vector<std::vector<float>>>& content);
    Input(const std::vector<std::vector<std::vector<std::vector<float>>>>& content);

    Input(std::initializer_list<float> content);
    Input(std::initializer_list<std::vector<float>> content);
    Input(std::initializer_list<std::vector<std::vector<float>>> content);
    Input(std::initializer_list<std::vector<std::vector<std::vector<float>>>> content);

    const Dtype& dtype() const;
    const Shape& shape() const;

    size_t rank() const;
    size_t size() const;

    void require() const;

    template <class T>
    T& at(size_t position);

  public:
    static Input fromObject(engine::Input* object);

  private:
    engine::Input* object() const;
    Input(engine::Input* object, char dummy);

    friend Stream& ::operator>>(matcha::Stream& stream, matcha::Input& input);
    friend std::ostream& ::operator<<(std::ostream& os, const Input& input);

    friend class Params;
    friend class Stream;
    friend class Tensor;
    friend class engine::Tensor;

  private:
    template <class NdimVector>
    std::vector<std::byte> buildBuffer(const NdimVector& content);

    void buildBuffer(std::vector<std::byte>& buffer, const float content);
    void buildBuffer(std::vector<std::byte>& buffer, const std::vector<float>& content);
    void buildBuffer(std::vector<std::byte>& buffer, const std::vector<std::vector<float>>& content);
    void buildBuffer(std::vector<std::byte>& buffer, const std::vector<std::vector<std::vector<float>>>& content);
    void buildBuffer(std::vector<std::byte>& buffer, const std::vector<std::vector<std::vector<std::vector<float>>>>& content);

    template <class NdimVector>
    Shape buildShape(const NdimVector& content);

    void buildShape(std::vector<unsigned>& axes, const float content);
    void buildShape(std::vector<unsigned>& axes, const std::vector<float>& content);
    void buildShape(std::vector<unsigned>& axes, const std::vector<std::vector<float>>& content);
    void buildShape(std::vector<unsigned>& axes, const std::vector<std::vector<std::vector<float>>>& content);
    void buildShape(std::vector<unsigned>& axes, const std::vector<std::vector<std::vector<std::vector<float>>>>& content);
};

Input floats(const Shape& shape);
Input constant(const Shape& shape, float value);
Input zeros(const Shape& shape);
Input ones(const Shape& shape);
Input eye(unsigned side);


}
