#include <iostream>
#include <matcha/tensor>
#include <matcha/engine>

using namespace matcha;
using namespace matcha::fn;
using namespace std;



void exportFlow() {
  Input i = eye(5);
  i.at<float>(3) = 9;
  Tensor t = exp(equal(matmul(eye(5), i) / .4, 22.5));
//  t.use(device::Cpu());
  Flow f(t);
  f.save("flow_of.matcha");
}

void importFlow() {
  Flow f = Flow::load("flow_of.matcha");

  Stream rand = rng::normal();
  f.test(rand);
}

void test() {
//  Stream mnist = dataset::csv("/home/patz/Downloads/mnist_train2.csv");

//  Model ai {
//    NeuralNetwork {
//      Initialization(xavier()),
//      Topology {
//        Affine(200),
//        Relu(),
//        Softmax()
//      }
//    }
//  };

//  ai.train(mnist);
}

Tensor softmax(const Tensor& a) {
  Tensor normed = a - max(a);
  Tensor numerator = exp(normed);
  Tensor denominator = sum(numerator);
  return numerator / denominator;
}

Tensor relu(const Tensor& a) {
  return max(a, 0);
}

Tuple msd(Stream& stream) {
  Tensor m = fn::fold(stream, 0, fn::add) / stream.size();
  Stream squares = fn::map(stream, [&](auto& t) {
    return fn::square(t - m);
  });
  Stream sd = fn::fold(squares, 0, fn::add) / (squares.size() - 1);
  return {m, sd};
}

int main() {
  Stream s = rng::normal(1, 1);
  Stream f = fn::map(s, [](auto& t) { return square(t) + 100; });
  Stream g = fn::map(f, [](auto& t) { return 0 - 10 * sqrt(t); });
  Stream h = fn::batch(g, 3000);
  Stream i = fn::batch(h, 2000);
  cout << fn::fold(i, 0, fn::add) / i.size();


  return 0;
}















