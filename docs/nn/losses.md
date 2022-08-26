# Loss functions

Loss functions set quantitative goals for artificial neural networks 
and tell them how close or far they are. They are binary operations
accepting a batch of `expected` results and `predicted` ones respectively,
and returning a batch of _losses_, representing how far the neural network
is from the "ideal" state of returning precisely what we want it to
return (note that this is a simplification). We want to minimize the
numbers our loss function is returning. For that, the loss function
**must be differentiable** (of course, you don't have to calculate the
derivatives, the [automatic differentiation engine](tensor/autograd)
will do that for you).

Feel free to implement your own loss function. 
Matcha implements the commonly used losses for regression and classification:

## Regression
> `mse(const tensor& expected, const tensor& predicted) -> tensor`

For regression, we usually want to use the 
[Mean Squared Error (MSE)](tensor/operations/miscellaneous#mse) function:

$ mse(a, b) = \frac{1}{|a|} \sum_{(i, j) \in (a, b)}(j - i)^2 $

It can be used as a loss, as it is indeed differentiable w.r.t. the
`predicted` argument ($ b $):

$ 
\frac{\partial mse(a, b)}{\partial b} =
\frac{1}{\partial b} \frac{1}{|a|} \sum_{(i, j) \in (a, b)}(j - i)^2 =
\frac{2 \cdot b}{|a|}
$

## Classification
> `class nn::Nll`

For classification tasks, `nn::Nll` wraps various cross-entropy functions.
It deduces which one to apply based on its input frames. For example,
for `predicted := Float[10]` and `expected := Int[]`, the sparse
categorical crossentropy would be applied 
(see e.g. [`tf.keras.losses.SparseCategoricalCrossentropy`](https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy)
and [`torch.nn.NLLLoss`](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html)).

In the case of the categorical cross entropy, the results are computed as:

$ H(a, b) = -log( \prod_{(i, j) \in (a, b)} j^i ) $

In the sparse variant (exactly one `expected` is one, others are zero),
the respective derivative of `predicted` is then:

$
\frac{\partial H(a, b)}{\partial b} = -
\frac{1}{\partial b} log( \prod_{(i, j) \in (a, b)} j^i ) = -
\frac{[a == 1]}{ \prod_{(i, j) \in (a, b)} j^i }
$
