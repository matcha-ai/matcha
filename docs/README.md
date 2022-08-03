
# Matcha

[Tensor arithmetic](tensor/) |
[Automatic differentiation](installation) |
[JIT compilation](tensor) |
[Dataset pipelines](dataset) |
[Neural networks](autograd) |
[Gotchas](gotchas)


## What it is

Matcha is a framework for optimized tensor arithmetic and
machine learning. It features a very intuitive interface
inspired by Numpy and Keras. Matcha brings all this to C++.
It also provides a way to accelerate itself by Just-In-Time
inspecting and modifying the structure of given tensor functions
and compiling them into a chain of instructions. On top of that,
Matcha delivers seamless dataset pipelines, automatic differentiation
system, and neural networks.

```cpp
#include <iostream>
#include <matcha>

using namespace matcha;

int main() {
  Net net {
    nn::flatten,                             // flatten the inputs
    nn::Fc{300, "relu,batchnorm"},           // hidden layer
    nn::Fc{10, "softmax"}                    // output layer
  };

  Dataset mnist = load("mnist_train.csv");   // load the MNIST dataset
  net.loss = nn::Nll{};                      // use the negative log likelihood loss
  net.fit(mnist.batch(64));                  // fit the model

  tensor digit = load("digit.png");          // load a single digit image
  tensor probabilities = net(digit);         // make a prediction
  tensor result = argmax(probabilities);     // voila

  std::cout << "it is " << result << " with "
            << probabilities[result] * 100 << "% probability" << std::endl;
}
```

For more examples, check out [this page](examples.md).


## Plans <small>(and what's currently missing)</small>

- More thorough CMake integration
- Python interface, Java/Kotlin interface
- Differentiable and JIT-compilable conditions and loops
- GPU acceleration (OpenCL, then CUDA)


## Troubleshooting

Please, open an issue under the appropriate package on [GitHub](https://github.com/matcha-ai).


## Contribute

See you on [GitHub](https://github.com/matcha-ai/)!

