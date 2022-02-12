#include "bits_of_matcha/input.h"

#include <matcha/engine>
#include <stdexcept>


namespace matcha {

Input::Input(const Dtype& dtype, const Shape& shape)
  : Object(new engine::Input(dtype, shape))
{}

Input::Input(const Stream& stream) {
  auto* tensor = stream.object()->open();
  auto* input  = new engine::Input(tensor->dtype(), tensor->shape());
  input->update(tensor);
  tensor->prune();
  reset(input);
}

Input::Input(float scalar)
  : Object(new engine::Input(Dtype::Float, {}, buildBuffer(std::vector<float>{scalar})))
{}

Input::Input(const std::vector<float>& content)
  : Object(new engine::Input(Dtype::Float, {buildShape(content)}, buildBuffer(content)))
{}

Input::Input(const std::vector<std::vector<float>>& content)
  : Object(new engine::Input(Dtype::Float, {buildShape(content)}, buildBuffer(content)))
{}

Input::Input(const std::vector<std::vector<std::vector<float>>>& content)
  : Object(new engine::Input(Dtype::Float, {buildShape(content)}, buildBuffer(content)))
{}

Input::Input(const std::vector<std::vector<std::vector<std::vector<float>>>>& content)
  : Object(new engine::Input(Dtype::Float, {buildShape(content)}, buildBuffer(content)))
{}

Input::Input(std::initializer_list<float> content)
  : Object(new engine::Input(Dtype::Float, {buildShape(content)}, buildBuffer(content)))
{}

Input::Input(std::initializer_list<std::vector<float>> content)
  : Object(new engine::Input(Dtype::Float, {buildShape(content)}, buildBuffer(content)))
{}

Input::Input(std::initializer_list<std::vector<std::vector<float>>> content)
  : Object(new engine::Input(Dtype::Float, {buildShape(content)}, buildBuffer(content)))
{}

Input::Input(std::initializer_list<std::vector<std::vector<std::vector<float>>>> content)
  : Object(new engine::Input(Dtype::Float, {buildShape(content)}, buildBuffer(content)))
{}

template <class NdimVector>
std::vector<std::byte> Input::buildBuffer(const NdimVector& content) {
  std::vector<std::byte> buffer;
  buildBuffer(buffer, content);
  return buffer;
}

template <class NdimVector>
Shape Input::buildShape(const NdimVector& content) {
  std::vector<unsigned> axes;
  buildShape(axes, content);
  return axes;
}

void Input::buildBuffer(std::vector<std::byte> &buffer, const float content) {
  auto begin = (std::byte*)(&content);
  auto end   = (std::byte*)(&content + 1);
  for (auto it = begin; it != end; it++) {
    buffer.push_back(*it);
  }
}

void Input::buildBuffer(std::vector<std::byte> &buffer, const std::vector<float>& content) {
  if (content.size() == 0) throw std::invalid_argument("empty value block");
  auto begin = (std::byte*)(&*content.begin());
  auto end   = (std::byte*)(&*content.end());
  for (auto it = begin; it != end; it++) {
    buffer.push_back(*it);
  }
}

void Input::buildBuffer(std::vector<std::byte> &buffer, const std::vector<std::vector<float>>& content) {
  if (content.size() == 0) throw std::invalid_argument("empty value block");
  size_t dim = content[0].size();
  for (auto& block: content) {
    if (block.size() != dim) throw std::invalid_argument("value block is ragged");
    buildBuffer(buffer, block);
  }
}

void Input::buildBuffer(std::vector<std::byte> &buffer, const std::vector<std::vector<std::vector<float>>>& content) {
  if (content.size() == 0) throw std::invalid_argument("empty value block");
  size_t dim = content[0].size();
  for (auto& block: content) {
    if (block.size() != dim) throw std::invalid_argument("value block is ragged");
    buildBuffer(buffer, block);
  }
}

void Input::buildBuffer(std::vector<std::byte> &buffer, const std::vector<std::vector<std::vector<std::vector<float>>>>& content) {
  if (content.size() == 0) throw std::invalid_argument("empty value block");
  size_t dim = content[0].size();
  for (auto& block: content) {
    if (block.size() != dim) throw std::invalid_argument("value block is ragged");
    buildBuffer(buffer, block);
  }
}

void Input::buildShape(std::vector<unsigned int> &axes, const float content) {
}

void Input::buildShape(std::vector<unsigned int> &axes, const std::vector<float>& content) {
  axes.push_back(content.size());
  buildShape(axes, *content.begin());
}

void Input::buildShape(std::vector<unsigned int> &axes, const std::vector<std::vector<float>>& content) {
  axes.push_back(content.size());
  buildShape(axes, *content.begin());
}

void Input::buildShape(std::vector<unsigned int> &axes, const std::vector<std::vector<std::vector<float>>>& content) {
  axes.push_back(content.size());
  buildShape(axes, *content.begin());
}

void Input::buildShape(std::vector<unsigned int> &axes, const std::vector<std::vector<std::vector<std::vector<float>>>>& content) {
  axes.push_back(content.size());
  buildShape(axes, *content.begin());
}

const Dtype& Input::dtype() const {
  if (isNull()) throw std::runtime_error("Ojbect is null");
  return object()->dtype();
}

const Shape& Input::shape() const {
  if (isNull()) throw std::runtime_error("Ojbect is null");
  return object()->shape();
}

void Input::update(const Tensor& source) {
  if (isNull() || source.isNull()) throw std::runtime_error("Ojbect is null");
  object()->update(source.object());
}

void Input::update() const {
  if (isNull()) throw std::runtime_error("Ojbect is null");
  object()->updateStatusChanged();
}

template <class T>
T& Input::at(size_t position) {
  if (isNull()) throw std::runtime_error("Ojbect is null");
  return object()->at<T>(position);
}

template float& Input::at(size_t);

engine::Input* Input::object() const {
  return reinterpret_cast<engine::Input*>(Object::object());
}

Input Input::fromObject(engine::Input* object) {
  return Input(object, 0);
}

Input::Input(engine::Input* object, char dummy)
  : Object(object)
{}

Input constant(const Shape& shape, float value) {
  std::vector<float> floatData(shape.size(), value);
  auto* floatBuffer = floatData.data();
  std::byte* byteBuffer = reinterpret_cast<std::byte*>(floatBuffer);
  std::vector<std::byte> byteData(byteBuffer, byteBuffer + sizeof(float) * floatData.size());
  auto* obj = new engine::Input(Dtype::Float, shape, byteData);
  return Input::fromObject(obj);
}

Input zeros(const Shape& shape) {
  return constant(shape, 0);
}

Input ones(const Shape& shape) {
  return constant(shape, 1);
}

Input eye(unsigned side) {
  Input a = zeros({side, side});
  for (unsigned i = 0; i < side; i++) {
    a.at<float>(i * (side + 1)) = 1;
  }
  return a;
}

}

std::ostream& operator<<(std::ostream& os, const matcha::Input& input) {
  auto& shape = input.shape();
  auto& dtype = input.dtype();

  os << "Input { "
     << "dtype: " << dtype.string() << ", "
     << "shape: [";

  for (int i = 0; i < shape.rank(); i++) {
    if (i != 0) os << ", ";
    os << shape[i];
  }
  os << "] }" << std::endl;
  return os;
}

