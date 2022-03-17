#include <matcha/matcha>
#include <iostream>

#include <any>
#include <vector>
#include <tuple>

struct CustomLayer {

};


void test() {
  ma::dataset::Csv mnist{
    .file = "/home/patz/Downloads/mnist_train2.csv",
    .delimiter = ';',
    .y_cols = {"label"}
  };
  print(mnist.size());

  nn::Sequential seq {
    [](tensor x) { return x / 256; },
    fn::exp,
    nn::Dense{ .units = 100, .use_bias = true, .activation = "relu" },
    fn::abs,
    nn::Affine{ .units = 100 },
    nn::Activation{ .activation = "tanh" },
    [](tensor x) { return 2 * x; },
    fn::argmax
  };

  seq.solver = nn::SGD {
    .learning_rate = .5,
    .loss = nn::CategoricalCrossentropy(),
    .epochs = 10,
    .batch_size = 32,
  };

  seq.fit(mnist);

}

