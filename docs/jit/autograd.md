# Autograd

> Out-of-the-box automatic differentiation of Flow

[Liquefied](flow/) functions can be easily made to calculate the first derivatives of specified variables.
This is done using the _backpropagation_ algorithm. In this article, you'll see how simple it is
to work with the Matcha Autograd. First things first, let's make a flow we'll work with:

```cpp
tensor params = uniform(5, 5);

auto foo = (Flow) [](tensor x) {
  return x * params;
};
```

We have created a simple Flow that takes external matrix `params`, multiplies it by the input `x`, and returns it.
Let's say we would like to see how that output changes with respect to ("w.r.t.") the matrix. Then we simply say:

```cpp
foo.requireGrad(&params);
```

From now on, the Flow will be able to compute the gradient of its output w.r.t to `params`. To do that,
run the Flow normally first (to let Matcha prepare the gradient propagation). Then start the gradient
computation:

```cpp
tensor output = foo(42);

for (auto [var, delta]: foo.grad()) {
  std::cout << "the derivative w.r.t. " << *var << " is " << delta;
}
```

?> Note that if the Flow outputs a tuple, the gradient is propagated back from its **first element**.

The backpropagation makes use of the well-known chain rule. It goes from the back and calculates deeper
and deeper partials, until it covers all the required gradients. 

Not all operations are necessarily differentiable though!
Take `abs`, which has no derivative at `0`. Even though it's not mathematically kosher, it's very practical to have such
points defined in some way, instead of throwing an exception right away and blocking the entire computation. For this reason,
the derivative of `abs(0)` returns 0. Another notable case is the Rectified Linear Unit, a.k.a. ReLU.

!> In other cases, differentiation is altogether **nonsensical**. \
This is usually the case when working with discrete values,
such as in `argmax`. If the derivative is not defined (or the Op simply doesn't implement it), it is assumed to be zero.
This will again make it possible to acquire at least some gradients. Moreover, even the locally unreachable gradients will be
computed with at least partial correctness, if there exists some other derivative chain leading to it.

## requireGrad

> `Flow::requireGrad(const tensor* wrt)` \
> `Flow::requireGrad(const std::vector<tensor*>& wrts)`

Requires gradient calculation w.r.t. given tensor(s).

## unrequireGrad

> `Flow::unrequireGrad(const tensor& wrt)` \
> `Flow::unrequireGrad(const std::vector<tensor*>& wrts)`

Unrequires gradient calculation w.r.t. given tensor(s).

## requiredGrad

> `Flow::requiredGrad()  ->  tuple`

Lists tensors with required gradient computation.

## setRequiredGrad

> `Flow::setRequiredGrad(const std::vector<tensor*>& wrts)`

Requires gradient computation only for the given list of tensors.

## grad

> `Flow::grad(tensor delta = 1)  ->  std::vector<std::pair<tensor, tensor>>`

Fires the backpropagation towards all the required gradient tensors. The forward_ run must be invoked first.
