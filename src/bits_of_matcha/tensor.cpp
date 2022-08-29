#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/ops.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/engine/tensor/Binding.h"
#include "bits_of_matcha/View.h"
#include "bits_of_matcha/engine/tensor/factories.h"
#include "bits_of_matcha/print.h"
#include "bits_of_matcha/engine/ops/Print.h"
#include "bits_of_matcha/engine/ops/SaveImage.h"
#include "bits_of_matcha/engine/ops/SaveCsv.h"
#include "bits_of_matcha/engine/lambda/Tracer.h"
#include "bits_of_matcha/engine/ops/Assign.h"

#include <filesystem>


using namespace matcha::engine;

namespace matcha {

tensor::tensor() noexcept
  : tensor(new Tensor({}))
{}

tensor::tensor(float scalar)
  : tensor(engine::full(scalar, {}))
{}

tensor::tensor(double scalar)
  : tensor(engine::full((float) scalar, {}))
{}

tensor::tensor(int8_t scalar)
  : tensor(engine::full(scalar, {}))
{}

tensor::tensor(int16_t scalar)
  : tensor(engine::full(scalar, {}))
{}

tensor::tensor(int32_t scalar)
  : tensor(engine::full(scalar, {}))
{}

tensor::tensor(int64_t scalar)
  : tensor(engine::full(scalar, {}))
{}

tensor::tensor(uint8_t scalar)
  : tensor(engine::full(scalar, {}))
{}

tensor::tensor(uint16_t scalar)
  : tensor(engine::full(scalar, {}))
{}

tensor::tensor(uint32_t scalar)
  : tensor(engine::full(scalar, {}))
{}

tensor::tensor(uint64_t scalar)
  : tensor(engine::full(scalar, {}))
{}

tensor::tensor(bool scalar)
  : tensor(engine::full(scalar, {}))
{}

tensor::tensor(std::complex<int32_t> scalar)
  : tensor(engine::full(scalar, {}))
{}

tensor::tensor(std::complex<uint32_t> scalar)
  : tensor(engine::full(scalar, {}))
{}

tensor::tensor(std::complex<float> scalar)
  : tensor(engine::full(scalar, {}))
{}

tensor::tensor(std::complex<double> scalar)
  : tensor(engine::full(std::complex<float>(scalar), {}))
{}

tensor& tensor::assign(const tensor& other) {
//  dispatch<ops::Assign>(deref(other), deref(this));
  return *this;
}

tensor& tensor::operator=(const tensor& other) {
  engine::Tensor* et = nullptr;
  if (internal_)
    et = ((engine::Binding*) internal_)->get();
  Tracer::handleNewRef(this, et);

  auto temp = identity(other);
  auto&& internal = (Binding*) internal_;
  if (internal) internal->unref();
  internal_ = temp.internal_;
  temp.internal_ = nullptr;
  return *this;
}

tensor& tensor::operator=(tensor&& other) noexcept {
  engine::Tensor* et = nullptr;
  if (internal_)
    et = ((engine::Binding*) internal_)->get();
  Tracer::handleNewRef(this, et);

  if (internal_ == other.internal_) return *this;
  auto&& internal = (Binding*) internal_;
  if (internal) internal->unref();
  internal_ = other.internal_;
  other.internal_ = nullptr;
  return *this;
}

tensor::tensor(const tensor& other) {
  Tracer::handleNewRef(this, nullptr);

  tensor copy = identity(other);
  internal_ = copy.internal_;
  copy.internal_ = nullptr;
}

tensor::tensor(tensor&& other) noexcept {
  Tracer::handleNewRef(this, nullptr);

  internal_ = other.internal_;
  other.internal_ = nullptr;
}

tensor::tensor(void* engineObject) {
  Tracer::handleNewRef(this, nullptr);

  auto&& internal = new Binding((Tensor*) engineObject);
  internal->ref();
  internal_ = internal;
}

tensor::~tensor() {
  auto&& internal = (Binding*) internal_;
  if (internal) {
    Tracer::handleDelRef(this, ((engine::Binding*) internal_)->get());
    internal->unref();
  } else {
    Tracer::handleDelRef(this, nullptr);
  }
}

const Frame& tensor::frame() const {
  return deref(this)->frame();
}

const Dtype& tensor::dtype() const {
  return frame().dtype();
}

const Shape& tensor::shape() const {
  return frame().shape();
}

size_t tensor::size() const {
  return shape().size();
}

size_t tensor::rank() const {
  return shape().rank();
}

tensor tensor::reshape(const Shape::Reshape& dims) const {
  return matcha::reshape(*this, dims);
}

tensor tensor::transpose() const {
  return matcha::transpose(*this);
}

tensor tensor::t() const {
  return matcha::transpose(*this);
}

tensor tensor::matmul(const tensor& b) const {
  return matcha::matmul(*this, b);
}

tensor tensor::power(const tensor& b) const {
  return matcha::power(*this, b);
}

tensor tensor::cast(const Dtype& dtype) const {
  return matcha::cast(*this, dtype);
}

auto tensor::operator[](const tensor& idxs) -> View {
  return ref(new engine::View((engine::Binding*) internal_, deref(idxs)));
}

auto tensor::operator[](const tensor& idxs) const -> View const {
  return ref(new engine::View((engine::Binding*) internal_, deref(idxs)));
}

void* tensor::data() {
  return deref(this)->readData();
}

tensor::operator bool() const {
  if (size() != 1)
    throw std::runtime_error("can't convert a non-scalar tensor into a single bool; use all(tensor) or any(tensor)");

  if (dtype() != Bool)
    return this->cast(Bool).operator bool();

  auto internal = deref(this);
  if (!internal->buffer())
    throw std::runtime_error("buffer is null");

  return internal->buffer().as<bool*>()[0];
}

void tensor::save(const std::string& file, SaveSpec spec) {
  std::filesystem::path path(file);
  std::string ext = path.extension();
  std::transform(ext.begin(), ext.end(), ext.begin(), tolower);
  engine::Op* op = nullptr;
  if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
    op = new engine::ops::SaveImage(deref(this), file);
  } else if (ext == ".csv") {
    op = new engine::ops::SaveCsv(deref(this), file);
  } else {
    throw std::runtime_error("unsupported export format: " + ext);
  }
  engine::dispatch(op);
}


tensor tensor::full(float value, const Shape& shape) {
  return ref(engine::full(value, shape));
}

tensor tensor::zeros(const Shape& shape) {
  return ref(engine::zeros(shape));
}

tensor tensor::ones(const Shape& shape) {
  return ref(engine::ones(shape));
}

tensor tensor::eye(const Shape& shape) {
  return ref(engine::eye(shape));
}

tensor tensor::blob(const void* data, const Frame& frame) {
  return ref(engine::blob(data, frame));
}

tensor tensor::blob(const void* data, const Dtype& dtype, const Shape& shape) {
  return ref(engine::blob(data, Frame{dtype, shape}));
}

tensor tensor::blob(const float* data, const Shape& shape) {
  return ref(engine::blob((void*) data, Frame{Float, shape}));
}

tensor tensor::blob(const std::vector<float>& data, const Shape& shape) {
  return blob(data.data(), shape);
}

tensor tensor::blob(const std::vector<float>& data) {
  return blob(data, {(unsigned) data.size()});
}

template <class Ilist>
void ilistDims(std::vector<unsigned>& dims, const Ilist& ilist) {
  if (ilist.size() == 0)
    throw std::runtime_error("tensor initialization data is empty");

  dims.push_back(ilist.size());

  if constexpr (std::is_convertible<Ilist, std::initializer_list<float>>()) {
  } else if constexpr (std::is_convertible<Ilist, float>()) {
  } else {

    auto&& first = *ilist.begin();
    ilistDims(dims, first);
  }
}

template <class Ilist>
void fromIlistInternal(std::vector<float>& buffer, const Ilist& ilist) {
  if constexpr (std::is_convertible<Ilist, std::initializer_list<float>>()) {
    for (auto&& val: ilist)
      buffer.push_back(val);

  } else if constexpr (std::is_convertible<Ilist, float>()) {
  } else {
    auto&& first = *ilist.begin();
    for (auto&& sublist: ilist) {
      if (sublist.size() != first.size())
        throw std::invalid_argument("tensor initialization data is ragged");
    }

    for (auto& sublist: ilist)
      fromIlistInternal(buffer, sublist);
  }
}

template <class Ilist>
inline engine::Tensor* fromIlist(Ilist& ilist) {
  std::vector<float> buffer;
  std::vector<unsigned> dims;
  ilistDims(dims, ilist);
  fromIlistInternal(buffer, ilist);
  return engine::blob(buffer.data(), Frame(Float, dims));
}

tensor::tensor(std::initializer_list<float> vector)
  : tensor(fromIlist(vector))
{}

tensor::tensor(std::initializer_list<std::initializer_list<float>> matrix)
  : tensor(fromIlist(matrix))
{}

tensor::tensor(std::initializer_list<std::initializer_list<std::initializer_list<float>>> rank3tensor)
  : tensor(fromIlist(rank3tensor))
{}

tensor::tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<float>>>> rank4tensor)
: tensor(fromIlist(rank4tensor))
{}

std::ostream& operator<<(std::ostream& os, const matcha::tensor& t) {
  auto op = new ops::Print(deref(t), false, os);
  op->init();
  op->run();
  delete op;
  return os;
}

}

