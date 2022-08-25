
# Matcha

[Tensor arithmetic](tensor/) |
[Automatic differentiation](tensor/autograd) |
[JIT compilation](tensor/jit) |
[Dataset pipelines](dataset/) |
[Neural networks](nn/net) |
[Gotchas](gotchas/)


## What it is

Matcha is a framework for optimized tensor arithmetic and
machine learning. It features a very intuitive interface
inspired by Numpy and Keras. Matcha brings all this to C++.
It also provides a way to accelerate itself by Just-In-Time
inspecting and modifying the structure of given tensor functions
and compiling them into a set of instructions. On top of that,
Matcha delivers a seamless dataset pipeline system, 
automatic differentiation system, and neural networks framework.

```cpp
#include <iostream>
#include <matcha>

using namespace matcha;

int main() {
  Net net {
    nn::flatten,                             // inlineExpansion the inputs
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

For more, check out [tutorials](tutorials/).

## License

Matcha is open source. It is available under the MIT license. 
It may be freely used and distributed.

## Plans <small>(and what's currently missing)</small>

- More thorough CMake integration
- Python interface, Java/Kotlin interface
- Differentiable and JIT-compilable conditions and loops
- GPU acceleration (OpenCL, then CUDA)


## Troubleshooting

Please, open an issue under the appropriate package on [GitHub](https://github.com/matcha-ai).


## Contribute

See you on [GitHub](https://github.com/matcha-ai/)!

