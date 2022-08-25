# Artificial Neural Networks
> `class Net`

Matcha `nn` module implements common concepts used in artificial neural 
network machine learning. This includes [`Layers`](nn/layers),
[`Losses`](nn/losses), and [`Optimizers`](nn/optimizers).
They can be assembled together to create fully functional machine learning
models. The class `Net` provides easy-to-use APIs for work with neural nets,
inspired by [Keras](https://keras.io/) and [PyTorch](https://pytorch.org/):

- Sequential API
- Subclassing API
- Functional API

## Sequential API

> `Net::Net(std::initializer_list<unary_fn> layers)` \
> `Net::Net(const std::vector<unary_fn>& layers)`


Sequential API is the most straightforward one. It lets you build 
a neural net simply by declaring its layers in a single list:

```cpp
Net net {
  nn::flatten,               // flatten the inputs
  nn::Fc{100, "tanh"},       // one hidden tanh layer
  nn::Fc{1, "sigmoid"}       // binary classification output layer
};
```

## Subclassing API

Subclassing API, on the other hand, leaves you the most flexibility.

## Functional API


!> This simplicity comes at a price. 
   The sequential API can only be used  to build nets with sequential
   topology. For more complex networks (e.g. with residual connections),
   use the functional or subclassing API.

## Training neural networks
