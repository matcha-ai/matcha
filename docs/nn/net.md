# Artificial Neural Networks
> `class Net`

Matcha `nn` module implements common concepts used in artificial neural 
network machine learning. This includes [`Layers`](nn/layers),
[`Losses`](nn/losses), and [`Optimizers`](nn/optimizers).
They can be assembled together to create fully functional machine learning
models. The class `Net` provides easy-to-use APIs for work with neural nets,
inspired by popular state-of-the-art frameworks like
[Keras](https://keras.io/) and [PyTorch](https://pytorch.org/):

- [Sequential](#sequential-api) API
- [Subclassing](#subclassing-api) API
- [Functional](#functional-api) API

After demonstrating each of these APIs, we will go through
[training](#training-neural-networks) neural networks and using them for
generating [predictions](#neural-network-predictions). 
Note however, that this guide is concerned with explaining the interface
and does not go into detail on _how to design neural networks_,
which can be found in [tutorials](tutorials/) (note: work in progress).

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

Done! Now we can [train](#training-neural-networks) it.


!> This simplicity comes at a price. 
   The sequential API can only be used  to build nets with sequential
   topology. For more complex networks (e.g. with _residual connections_),
   use the functional or subclassing API.

## Subclassing API

Subclassing API, on the other hand, leaves you the most flexibility.
The trade-off for that is the most extra code. It works through inheriting
`Net` and overriding its protected virtual logic:

> `virtual Net::run(const tensor& a) -> tensor` \
> `virtual Net::run(const tensor& a, const tensor& b) -> tensor` \
> `virtual Net::run(const tensor& a, const tensor& b, const tensor& c) -> tensor` \
> `virtual Net::run(const tuple& inputs) -> tuple`

Single batch processing function.

> `virtual Net::init(const tensor& a) -> void` \
> `virtual Net::init(const tensor& a, const tensor& b) -> void` \
> `virtual Net::init(const tensor& a, const tensor& b, const tensor& c) -> void` \
> `virtual Net::init(const tuple& inputs) -> void`

Single batch processing function initialization - invoked exactly once,
before the first `Net::run` call. Accepts the same arguments as the
invoked `Net::run`.

> `virtual Net::trainStep(Instance i) -> void`

Customizable train step logic. By default, it performs one _forward_ and
_backward_ propagation using [`Backprop`](tensor/autograd#backprop), 
emitting appropriate [callback signals](nn/callbacks/signals).

!> In contrast to static machine learning frameworks like 
   [TensorFlow](https://www.tensorflow.org/), which let you merely
   _declare_ the network topology through enumerating its components,
   Matcha allows more flexibility through dynamic flow. However, this
   means that all components that the neural net uses must be stored
   somethere so that they can be explicitly called later in your code.
   For example, instantiating and calling a layer all at once inside
   the `run` method **would not work as expected in TensorFlow**. Instead,
   a new layer would be created in each `run` invokation. Therefore,
   **you must instantiate all layers before calling `run`**, e.g. from `init`
   as private class members.

An example will follow. We will create a custom `FcResNet` class using the
`Net` subclassing API. To demonstrate the flexibility of the subclassing
API, the `FcResNet` class will implement automatic _residual connection_
creation logic on fully connected (`nn::Fc`) layers:

```cpp
class FcResNet : public Net {
  auto preprocessor = nn::Fc{100, "relu"};
  std::vector<unary_fn> residual_blocks;
  auto output = nn::Fc{10, "softmax"};

  void createResBlock();
  void init(const tensor& input) override;
  tensor run(const tensor& input) override;
};
```

Now, let's implement the methods. We will start by `createResBlock` logic.
There are many ways to do this. We will be using a value-capturing C++ lambda:

```cpp
void FcResNet::createResBlock() {
  unary_fn block = [
                      fc1 = nn::Fc{100, "relu"},
                      fc2 = nn::Fc{200, "relu"},
                      fc3 = nn::Fc{100, "none"}
                   ]
                   (const tensor& a) {
                     return fc3(fc2(fc1(a))) + a;
                   };

  residual_blocks.emplace_back(std::move(block));
}
```

Now, we will define the `init` function. We will let it create 3 residual blocks:

```cpp
void FcResNet::init(const tensor& input) {
  for (int i = 0; i < 3; i++)
    createResBlock();
}
```

And at last, we create the `run` function. Since we've done most of the
hard work in `createResBlock` and `init`, this function can merely 
sequentially call the stored residual blocks, with some pre-processing
and post-processing:

```cpp
tensor FcResNet::run(const tensor& input) {
  tensor feed = preprocessor(input);
  for (auto& block: residual_blocks)
    feed = block(feed);
  
  return output(feed);
}
```

That's it! We can proceed to instantiating our `FcResNet`:

```cpp
FcResNet net;
```

... and [training](#training-neural-networks) it.

## Functional API

> `Net::Net(const fn& function)`

The functional API is midway between the sequential and the subclassing API.
Use it when the network you want to create does not have a sequential 
topology but is still small enough. This can be done using a lambda or
by wrapping a normal function. We will make use of a C++ lambda with
C++ _static variables_ storing our layers (remember, Matcha is not _static_ itself!)
to create a simple net with one gated block, similar to those used in 
[recurrent neural networks](https://en.wikipedia.org/wiki/Recurrent_neural_network) (RNNs):

```cpp
Net net = (fn) [](tensor feed) {
  static auto fc1 = nn::Fc{100, "relu"};
  static auto fc2 = nn::Fc{100, "tanh"};
  static auto output = nn::Fc{10, "softmax"};

  feed = fc1(feed) * fc2(feed);
  return output(feed);
};
```

The network can now be [trained](#training-neural-neural).

## Training neural networks
> `Net::fit(Dataset ds, size_t epochs = 10) -> void` \
> `Net::epoch(Dataset ds) -> void` \
> `Net::step(Instance i) -> void`

Having our network logic declared, we can proceed to training it. 
We will to this in 4 steps.

#### Step 1: Choose a loss function

First, we must choose a [`Loss`](nn/losses) function for our network. 
Loss functions set quantitative goals for artificial neural networks and
tell them how close or far they are. Choose your loss function based on 
what you expect from the network. Common losses are:

- `mse` - Mean Squared Error loss for **regression**-based tasks
- `nn::Nll` - Negative Log Likelihood wrapping binary and categorical
   distribution cross-entropies for **classification**-based tasks

... or, create your own loss! In Matcha, this is as simple as defining
a normal binary function, the first argument being the batch of `expected` 
outputs, the second argument being the batch of `predicted` outputs. Note
however, that the loss function **must be differentiable**.

Let's suppose we want to train a regressive model:

```cpp
net.loss = mse;     
```

#### Step 2: Choose a neural network optimizer

The loss we have just chosen sets a goal for our neural network.
An `Optimizer` uses the gradients of that loss with respect to (w.r.t.)
our net's trainable parameters to minimize the loss and approach to 
the goal. By default, `Net` uses the stochastic
[Adaptive Moment Estimation (Adam)](https://arxiv.org/abs/1412.6980)
algorithm (`nn::Adam`), which has proven to be the most efficient for the 
vast majority of uses.
For this reason, we **can usually skip this step altogether**.

#### Step 3: Prepare a dataset

This step may equally important to designing the entire neural net.
It involves collecting data from the internet or otherwise, formatting
it, and assembling it into a single dataset. We will show here only
how to import an already prepared dataset and perform some pre-processing.
For a more, refer to the [dataset](dataset/) documentation section.

First, we have to load data from this disk. In this case, we will load the 
[Sklearn California housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html#sklearn.datasets.fetch_california_housing)
dataset from a `.csv` file. The dataset contains 20640 instances with
just 8 features and a single real-valued target for regression:

```cpp
Dataset california_housing = load("california_housing.csv");
std::cout << california_housing.size() << std::endl;            

// 20640
```

Next, we may want to make some adjustments to the dataset.
Notably, if every feature is represented as a separate scalar tensor,
we need to create a single input tensor by _mapping_ the dataset.
Dataset pipelines make it possible to work with huge amounts of data
(possibly billions of instances) in a memory-efficient manner:

```cpp
california_housing = california_housing.map([](Instance& i) {
  Instance mapped;
  mapped["y"] = i["Target"];
  mapped["x"] = stack(i["MedInc"],      // median income in block group
                      i["HouseAge"],    // median house age in block group
                      i["AveRooms"],    // average number of rooms per household
                      i["AveBedrms"],   // average number of bedrooms per household
                      i["Population"],  // block group population
                      i["AveOccup"],    // average number of household members
                      i["Latitude"],    // block group latitude
                      i["Longitude"]);  // block group longitude
  i = mapped;
});
```

Now when that's done, we have the training logic ready.

#### Step 4: Fit!

Finally, we can fit our model. By default, the fitting process will be
logged by the `nn::Logger` net callback. To disable it, simply clear
the net's callbacks:

```cpp
net.callbacks.clear();
```

Alternatively, you can add more callbacks. [Read more](net/callbacks).

To fit our dataset, simply:

```cpp
net.fit(california_housing);
```

If we have the `nn::Logger` enabled, the fitting process will be reported:

![img](fit.gif)

We can specify the number of epochs the fitting algorithm shall perform:

```cpp
int epochs = 3;
net.fit(california_housing, epochs);
```

Alternatively, we can perform just one epoch explicitly
(equivalent to `epochs = 1`):

```cpp
net.epoch(california_housing);
```

If we want the most control over iterating through the dataset instances,
we can even schedule each training step individually:

```cpp
for (int i = 0; i < california_housing.size(); i++) {
  Instance i = california_housing.get();
  net.step(i);
}
```

## Neural network predictions
> `operator()(const tensor& a) -> tensor` \
> `operator()(const tensor& a, const tensor& b) -> tensor` \
> `operator()(const tensor& a, const tensor& b, const tensor& c) -> tensor` \
> `operator()(const tuple& inputs) -> tuple`

If we have designed the network correctly, it will be able to 
_generalize_ what it has been trained. This means we can now use it to 
predict novel data. In this sense, the `Net` class behaves as a completely
normal function. It accepts a batch of input data and returns a batch
of respective predictions:

```cpp
tensor data = load("novel_housnig_context.csv");
tensor pred = net(data);

std::cout << "Neural Network predictions are:" << std::endl;
std::cout << pred << std::endl;
```

!> Note that the neural network accepts _a batch_ of inputs,
   not a _single_ input. Confusing these two can lead to errors
   or non-sensical results. To easily convert a single `input` to
   a single-input batch, use the `stack` operation. This will expand
   the input dimensionality by batch axis while retaining the 
   original shape: \
   `tensor batched_input = stack(input);`

