#include <matcha/matcha>
#include "/home/patz/benchmark/lib/benchmark.h"

using namespace matcha;

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


int main() {
  Benchmark benchmark("matcha", "/home/patz/benchmark/data");

  print("add");
  benchmark.run([] { run_op_tiny(add); }, 300, "add_tiny");
  benchmark.run([] { run_op_small(add); }, 100, "add_small");
  benchmark.run([] { run_op_small_scalar(add); }, 100, "add_small_scalar");
  benchmark.run([] { run_op_small_broadcast(add); }, 100, "add_small_broadcast");

  benchmark.run([] { run_op_big(add); }, 100, "add_big");
  benchmark.run([] { run_op_big_scalar(add); }, 100, "add_big_scalar");
  benchmark.run([] { run_op_big_broadcast(add); }, 100, "add_big_broadcast");

  benchmark.run([] { run_op_huge(add); }, 20, "add_huge");
  benchmark.run([] { run_op_huge_scalar(add); }, 20, "add_huge_scalar");
  benchmark.run([] { run_op_huge_broadcast(add); }, 20, "add_huge_broadcast");

  print("dot");
  benchmark.run([] { run_op_tiny(dot); }, 300, "dot_tiny");
  benchmark.run([] { run_op_small(dot); }, 100, "dot_small");
  benchmark.run([] { run_op_big(dot); }, 100, "dot_big");

  print("exp");
  benchmark.run([] { run_op_tiny(matcha::exp); }, 300, "exp_tiny");
  benchmark.run([] { run_op_small(matcha::exp); }, 100, "exp_small");
  benchmark.run([] { run_op_big(matcha::exp); }, 100, "exp_big");
  benchmark.run([] { run_op_huge(matcha::exp); }, 20, "exp_huge");
}