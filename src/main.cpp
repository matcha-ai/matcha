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

  Stream s = dataset::csv("/home/patz/Downloads/mnist_train2.csv");
  s = s.batch(100);
  Tensor it = s();
  Tensor x = it.reshape({28, 28});

  while (Stream batch = s.batch(5)) {
    cout << "begin batch" << std::endl;
    it.subst(batch);
    Tensor y = batch();
    while (batch.next()) {
      cout << x.plot() << std::endl;
      cout << "=> " << y.data().i() << std::endl;
    }
  }


  return 0;
}









