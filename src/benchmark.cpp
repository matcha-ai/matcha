#include <matcha>
#include "/home/patz/benchmark/lib/benchmark.h"

using namespace matcha;
using namespace std::complex_literals;

void net();
Dataset dataset();


auto fmean = (Flow) [](tensor a) {
  return mean(a);
};

auto info = (Flow) [](tensor a) {
//  return min(a) + fmean(a);
  return fmean(a);
};

tensor foo(tensor a) {
//  a = a.cast(Float);
  return info(a);// + fmean(a);
};


int main() {
//  net(); return 0;

  for (int i = 0; i < 5; i++) {
    i = 0;

    auto f = jit(foo);
//    print(info(ones(3, 3)));
    print(f(ones(3, 3)));
//    break;

  }
}





































Dataset dataset() {
  Dataset mnist = load("mnist_test.csv");
  mnist = mnist.take(1);
  for (int i = 0; i < 2; i++)
    mnist = mnist.cat(mnist);

  return mnist;
}

void net() {
  Net net {
  nn::Flatten{},
  nn::Fc{400},
  nn::Fc{100, "relu"},
  nn::Fc{50, "relu"},
  nn::Fc{10, "softmax"},
  };

  net.loss = mse;
//  net.callbacks.clear();

//  Dataset mnist = load("mnist_train.csv");
  Dataset mnist = (Dataset) []() {
    Instance i;
    i["x"] = ones(28, 28);
    i["y"] = cast(1, Int).reshape(1, 1);
    return i;
  };
  mnist = mnist.take(1000);
  mnist = mnist.cat(mnist).cat(mnist).cat(mnist);
  mnist = load("mnist_train.csv");
  net.fit(mnist.batch(30));
}

void run_op(const BinaryOp& op, const Shape& a, const Shape& b) {
  tensor ta = ones(a);
  tensor tb = ones(b);
  tensor tc = op(ta, tb);
}
void run_op(const UnaryOp& op, const Shape& a) {
  tensor ta = ones(a);
  tensor tc = op(ta);
}

void run_op_tiny(const BinaryOp& op) {
  run_op(op, {1, 1}, {1, 1});
}

void run_op_tiny(const UnaryOp& op) {
  run_op(op, {1, 1});
}

void run_op_small(const BinaryOp& op) {
  run_op(op, {7, 7}, {7, 7});
}

void run_op_small(const UnaryOp& op) {
  run_op(op, {7, 7});
}

void run_op_small_scalar(const BinaryOp& op) {
  run_op(op, {7, 7}, {});
}

void run_op_small_broadcast(const BinaryOp& op) {
  run_op(op, {7, 7}, {7, 1});
}

void run_op_big(const BinaryOp& op) {
  run_op(op, {1000, 1000}, {1000, 1000});
}

void run_op_big(const UnaryOp& op) {
  run_op(op, {1000, 1000});
}

void run_op_big_scalar(const BinaryOp& op) {
  run_op(op, {1000, 1000}, {});
}

void run_op_big_broadcast(const BinaryOp& op) {
  run_op(op, {1000, 1000}, {1000, 1});
}

void run_op_huge(const BinaryOp& op) {
  run_op(op, {10000, 10000}, {10000, 10000});
}

void run_op_huge(const UnaryOp& op) {
  run_op(op, {10000, 10000});
}

void run_op_huge_scalar(const BinaryOp& op) {
  run_op(op, {10000, 10000}, {});
}

void run_op_huge_broadcast(const BinaryOp& op) {
  run_op(op, {10000, 10000}, {10000, 1});
}

void run_op(const BinaryOp& op, size_t scale) {
  auto uscale = (unsigned) scale;
  run_op(op, {uscale}, {uscale});
}

void run_op(const UnaryOp& op, size_t scale) {
  auto uscale = (unsigned) scale;
  run_op(op, Shape{uscale});
}

void run_op_square(const BinaryOp& op, size_t scale) {
  auto uscale = (unsigned) scale;
  run_op(op, {uscale, uscale}, {uscale, uscale});
}



void benchmark() {
  Benchmark bm("matcha-OBLAS,march", "/home/patz/bm/data");
  bm.linspace([] (size_t scale){ run_op(add, scale); },
              1, 5'000'000, 100, 10, "add");

  bm.linspace([] (size_t scale){ run_op_square(dot, scale); },
              1, 3'000, 100, 10, "dot");

  bm.linspace([] (size_t scale){ run_op(matcha::exp, scale); },
              1, 5'000'000, 100, 10, "exp");

  return;
//  /*
  print("add");
  bm.run([] { run_op_tiny(add); }, 300, "add_tiny");
  bm.run([] { run_op_small(add); }, 100, "add_small");
  bm.run([] { run_op_small_scalar(add); }, 100, "add_small_scalar");
  bm.run([] { run_op_small_broadcast(add); }, 100, "add_small_broadcast");

  bm.run([] { run_op_big(add); }, 100, "add_big");
  bm.run([] { run_op_big_scalar(add); }, 100, "add_big_scalar");
  bm.run([] { run_op_big_broadcast(add); }, 100, "add_big_broadcast");

  bm.run([] { run_op_huge(add); }, 20, "add_huge");
  bm.run([] { run_op_huge_scalar(add); }, 20, "add_huge_scalar");
  bm.run([] { run_op_huge_broadcast(add); }, 20, "add_huge_broadcast");
//   */

  print("dot");
  bm.run([] { run_op_tiny(dot); }, 300, "dot_tiny");
  bm.run([] { run_op_small(dot); }, 100, "dot_small");
  bm.run([] { run_op_big(dot); }, 100, "dot_big");

//  /*
  print("exp");
  bm.run([] { run_op_tiny(matcha::exp); }, 300, "exp_tiny");
  bm.run([] { run_op_small(matcha::exp); }, 100, "exp_small");
  bm.run([] { run_op_big(matcha::exp); }, 100, "exp_big");
  bm.run([] { run_op_huge(matcha::exp); }, 20, "exp_huge");
//   */
}
