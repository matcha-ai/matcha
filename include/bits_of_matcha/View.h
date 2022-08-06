#pragma once

#include "bits_of_matcha/Frame.h"
#include "bits_of_matcha/tensor.h"


namespace matcha {

class tensor;

class View {
public:
  View(const View& other) = delete;
  View(View&& other) = delete;
  ~View();

  View& operator=(const View& other) = delete;
  View& operator=(View&& other) = delete;

  operator tensor() const;

  const Frame& frame() const;
  const Dtype& dtype() const;
  const Shape& shape() const;

  explicit operator bool() const;

  View& operator=(const tensor& t);

  auto operator[](const tensor& idx) -> View;
  auto operator[](const tensor& idx) const -> const View;

private:
  explicit View(void* internal);
  friend class Engine;
  void* internal_;
};

}