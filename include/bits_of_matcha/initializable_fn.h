#pragma once

#include "bits_of_matcha/fn.h"
#include "bits_of_matcha/tensor.h"

#include <cinttypes>


namespace matcha {

class InitializableUnaryFn {
public:
  InitializableUnaryFn();
  tensor operator()(const tensor& a);

protected:
  virtual void init(const tensor& a);
  virtual tensor run(const tensor& a) = 0;

private:
  bool initialized_;

};

class InitializableBinaryFn {
public:
  InitializableBinaryFn();
  tensor operator()(const tensor& a, const tensor& b);

protected:
  virtual void init(const tensor& a, const tensor& b);
  virtual tensor run(const tensor& a, const tensor& b) = 0;

private:
  bool initialized_;

};

class InitializableTernaryFn {
public:
  InitializableTernaryFn();
  tensor operator()(const tensor& a, const tensor& b, const tensor& c);

protected:
  virtual void init(const tensor& a, const tensor& b, const tensor& c);
  virtual tensor run(const tensor& a, const tensor& b, const tensor& c) = 0;

private:
  bool initialized_;

};

class InitializableNaryFn {
public:
  InitializableNaryFn();
  Tuple operator()(const Tuple& tuple);

protected:
  virtual void init(const Tuple& tuple);
  virtual Tuple run(const Tuple& tuple) = 0;

private:
  uint8_t ins_, outs_;

};

}