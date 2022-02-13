#include <iostream>
#include <matcha/tensor>
#include <matcha/dataset>
#include <matcha/model>
#include <matcha/nn>
#include <matcha/device>


using namespace std;
using namespace matcha;
using namespace matcha::nn;

//void nn() {
//  Context ctx;
//  ctx.debug(true);
//  Stream mnist = dataset::csv("/home/patz/Downloads/mnist_train2.csv");
//  mnist = fn::batch(mnist, 80);

//  Model ai {
//    NeuralNetwork {
//      Topology {
//        Affine(100),
//        Relu(),
//        Affine(10),
//        Softmax()
//      }
//    }
//  };

//  ai.train(mnist);
//}

Tuple msd(Stream& s) {
  Tensor m = s.fold(0, fn::add) / s.size();
  Stream squares = s.map([&](auto& x) { return fn::square(x - m); });
  Tensor sd = squares.fold(0, fn::add) / (squares.size() - 1);
  return {m, sd};
}

int main() {
  Context ctx;

  Stream s = dataset::csv("/home/patz/Downloads/mnist_train2.csv").map(fn::reshape({28, 28}));
//  s = s.map(fn::identity).batch(3);
  s = s.batch(10);
  Tuple x = msd(s);

  std::cout << (x[0] + x[1]).plt();

  s = s.batch(11);
  while (Stream batch = s.batch(5)) {
    Tensor a = s();
    while (batch.next()) {
      cout << a.plt() << std::endl;
    }
    std::cout << "batch done; size " << batch.size() << std::endl;;
  }


  return 0;
}









