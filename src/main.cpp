#include <iostream>
#include <matcha/tensor>
#include <matcha/engine>
#include <matcha/dataset>
#include <matcha/model>
#include <matcha/nn>

using namespace matcha;
using namespace matcha::fn;
using namespace matcha::nn;
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
  Stream mnist = dataset::csv("/home/patz/Downloads/mnist_train2.csv");

  Model ai {
    NeuralNetwork {
      Initialization(xavier()),
      Topology {
        Affine(200),
        Relu(),
        Softmax()
      }
    }
  };

  ai.train(mnist);
}

int main() {



  return 0;
}
