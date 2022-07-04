#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/ops.h"
#include "bits_of_matcha/engine/tensor/Tensor.h"
#include "bits_of_matcha/engine/tensor/factories.h"
#include "bits_of_matcha/print.h"
#include "bits_of_matcha/engine/ops/Print.h"
#include "bits_of_matcha/engine/ops/SaveImage.h"
#include "bits_of_matcha/engine/ops/SaveCsv.h"

#include <filesystem>


using namespace matcha::engine;

namespace matcha {

tensor::tensor() : internal_(new Tensor({})) {
  deref(this)->ref();
}

tensor::tensor(float scalar) : internal_(engine::full(scalar, {})) {
  deref(this)->ref();
}

tensor::tensor(double scalar) : internal_(engine::full(scalar, {})) {
  deref(this)->ref();
}

tensor::tensor(int8_t scalar) : internal_(engine::full(scalar, {})) {
  deref(this)->ref();
}

tensor::tensor(int16_t scalar) : internal_(engine::full(scalar, {})) {
  deref(this)->ref();
}

tensor::tensor(int32_t scalar) : internal_(engine::full(scalar, {})) {
  deref(this)->ref();
}

tensor::tensor(int64_t scalar) : internal_(engine::full(scalar, {})) {
  deref(this)->ref();
}

tensor::tensor(uint8_t scalar) : internal_(engine::full(scalar, {})) {
  deref(this)->ref();
}

tensor::tensor(uint16_t scalar) : internal_(engine::full(scalar, {})) {
  deref(this)->ref();
}

tensor::tensor(uint32_t scalar) : internal_(engine::full(scalar, {})) {
  deref(this)->ref();
}

tensor::tensor(uint64_t scalar) : internal_(engine::full(scalar, {})) {
  deref(this)->ref();
}

tensor::tensor(bool scalar) : internal_(engine::full(scalar, {})) {
  deref(this)->ref();
}

tensor::tensor(std::complex<int32_t> scalar) : internal_(engine::full(scalar, {})) {
  deref(this)->ref();
}

tensor::tensor(std::complex<uint32_t> scalar) : internal_(engine::full(scalar, {})) {
  deref(this)->ref();
}

tensor::tensor(std::complex<float> scalar) : internal_(engine::full(scalar, {})) {
  deref(this)->ref();
}

tensor::tensor(std::complex<double> scalar) : internal_(engine::full(scalar, {})) {
  deref(this)->ref();
}

tensor& tensor::operator=(const tensor& other) {
  auto temp = identity(other);
  if (internal_) deref(this)->unref();
  internal_ = unref(temp);
  return *this;
}

tensor& tensor::operator=(tensor&& other) {
  internal_ = other.internal_;
  other.internal_ = nullptr;
  return *this;
}

tensor::tensor(const tensor& other) {
  auto temp = identity(other);
  internal_ = unref(temp);
}

tensor::tensor(tensor&& other) noexcept {
  internal_ = other.internal_;
  other.internal_ = nullptr;
}

tensor::tensor(void* engineObject) {
  internal_ = engineObject;
  if (internal_) deref(this)->ref();
}

tensor::~tensor() {
  if (internal_) deref(this)->unref();
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

tensor tensor::dot(const tensor& b) const {
  return matcha::dot(*this, b);
}

tensor tensor::pow(const tensor& b) const {
  return matcha::pow(*this, b);
}

tensor tensor::cast(const Dtype& dtype) const {
  return matcha::cast(*this, dtype);
}

void* tensor::data() {
  return deref(this)->readData();
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
  engine::send(op);
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

}

std::ostream& operator<<(std::ostream& os, const matcha::tensor& t) {
  auto op = new ops::Print(deref(t), false, os);
  send(op);
  return os;
}