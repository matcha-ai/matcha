#include "bits_of_matcha/engine/ops/Pow.h"

#include <cmath>
#include <numeric>


namespace matcha::engine::ops {

Dtype promoteDtypesPow(Tensor* a, Tensor* b) {
  Dtype temp = promoteDtypes(a, b);
  if (temp == Bool) return Sbyte;
  return temp;
}

Pow::Pow(Tensor* a, Tensor* b)
  : ElementwiseBinaryOp(a, b, promoteDtypesPow(a, b))
{}

OpMeta<Pow> Pow::meta {
  .name = "Pow",
  .back = [](auto ctx) { return new PowBack(ctx); }
};


void Pow::run() {
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


PowBack::PowBack(const BackCtx& ctx)
  : OpBack(ctx)
{
}

OpMeta<PowBack> PowBack::meta {
  .name = "PowBack",
};

}
