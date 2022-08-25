#include "bits_of_matcha/engine/ops/Power.h"
#include "bits_of_matcha/engine/cpu/kernels/fill.h"
#include "bits_of_matcha/engine/cpu/kernels/elementwiseBinaryBack.h"

#include <cmath>
#include <numeric>


namespace matcha::engine::ops {

Dtype promoteDtypesPow(Tensor* a, Tensor* b) {
  Dtype temp = promoteDtypes(a, b);
  if (temp == Bool) return Sbyte;
  return temp;
}

Power::Power(Tensor* a, Tensor* b)
  : ElementwiseBinaryOp(a, b, promoteDtypesPow(a, b))
{}

Reflection<Power> Power::reflection {
  .name = "Power",
  .back = [](auto& ctx) { return dispatch<PowerBack>(ctx); },
};


void Power::run() {
  Dtype dtype = outputs[0]->dtype();
  auto a = inputs[0]->buffer();
  auto b = inputs[1]->buffer();
  auto c = outputs[0]->malloc();

  switch (dtype) {
  case Sbyte: cpu::elementwiseBinary<int8_t>([](auto a, auto b) { return std::pow(a, b);}, a, b, c, ctx_); break;
  case Short: cpu::elementwiseBinary<int16_t>([](auto a, auto b) { return std::pow(a, b);}, a, b, c, ctx_); break;
  case Int: cpu::elementwiseBinary<int32_t>([](auto a, auto b) { return std::pow(a, b);}, a, b, c, ctx_); break;
  case Long: cpu::elementwiseBinary<int64_t>([](auto a, auto b) { return std::pow(a, b);}, a, b, c, ctx_); break;

  case Byte: cpu::elementwiseBinary<uint8_t>([](auto a, auto b) { return std::pow(a, b);}, a, b, c, ctx_); break;
  case Ushort: cpu::elementwiseBinary<uint16_t>([](auto a, auto b) { return std::pow(a, b);}, a, b, c, ctx_); break;
  case Uint: cpu::elementwiseBinary<uint32_t>([](auto a, auto b) { return std::pow(a, b);}, a, b, c, ctx_); break;
  case Ulong: cpu::elementwiseBinary<uint64_t>([](auto a, auto b) { return std::pow(a, b);}, a, b, c, ctx_); break;

  case Float: cpu::elementwiseBinary<float>([](auto a, auto b) { return std::pow(a, b);}, a, b, c, ctx_); break;
  case Double: cpu::elementwiseBinary<double>([](auto a, auto b) { return std::pow(a, b);}, a, b, c, ctx_); break;

  case Cint: cpu::elementwiseBinary<std::complex<int32_t>>([](auto a, auto b) { return std::pow(a, b);}, a, b, c, ctx_); break;
  case Cuint: cpu::elementwiseBinary<std::complex<int64_t>>([](auto a, auto b) { return std::pow(a, b);}, a, b, c, ctx_); break;
  case Cfloat: cpu::elementwiseBinary<std::complex<float>>([](auto a, auto b) { return std::pow(a, b);}, a, b, c, ctx_); break;
  case Cdouble: cpu::elementwiseBinary<std::complex<double>>([](auto a, auto b) { return std::pow(a, b);}, a, b, c, ctx_); break;
  }
}


PowerBack::PowerBack(const BackCtx& ctx)
  : ElementwiseBinaryOpBack(ctx)
{
}

Reflection<PowerBack> PowerBack::reflection {
  .name = "PowerBack",
};

void PowerBack::run() {
  if (outputs[0]) {
    cpu::fill(outputs[0]->malloc(), outputs[0]->size(), (float) 0);

    float* begin_ga = outputs[0]->buffer().as<float*>();
    float* begin_a = forwardInput(0)->buffer().as<float*>();

    cpu::elementwiseBinaryBack(
    [=](float& ga, float& b, float& gc) {
      float a = begin_a[begin_ga - &ga];
//        std::cout << a << " " << b << " " << c << std::endl;
      // c = a^b
      // dc/da = b * a^(b-1)
      ga += b * pow(a, b - 1) * gc;
    },
    outputs[0]->buffer(),
    forwardInput(1)->buffer(),
    inputs[0]->buffer(),
    iter_
    );
  }
}

}
