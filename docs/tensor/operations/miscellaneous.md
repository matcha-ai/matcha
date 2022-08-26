# Miscellaneous operations

## matmul
> `matmul(const tensor& a, const tensor& b) -> tensor`

Matrix multiplication:

$ a \cdot b $

?> If the inputs have rank higher than two, `matmul` is performed matrix-wise;
   matrix-wise broadcasting is performed if necessary.

!> For backend upstream reasons, matrix multiplication is currently supported
   only for dtypes `Float` and `Double`.


## transpose
> `transpose(const tensor& a) -> tensor` \
> `tensor::transpose() const -> tensor` \
> `tensor::t() const -> tensor`

Matrix-wise transposition:

$ a^T $


## reshape
> `reshape(const tensor& a, const Shape::Reshape& dims) -> tensor` \
> `tensor::reshape(const Shape::Reshape& dims) -> tensor`

Reinterprets tensor shape without changing the order of contained elements.
The resulting tensor size must match the original tensor size. 

?> One dimension can be set to `-1`.
   Matcha will automatically infer its actual value.

## stack
> `stack(const std::vector<tensor>& tensors) -> tensor` \
> `template <class... Tensors> stack(Tensors... tensors) -> tensor`

Stacks tensors of the same shape along a new first axis.


## softmax
> `softmax(const tensor& x) -> tensor` \
> `softmax(const tensor& x, int axis) -> tensor`

Performs the softmax function. Summing the output tensor along the
same sequences (axis or globally) is approximately equal to `1`:

$ softmax(x) = \frac{e^{\odot \hat{x}}}{\sum_{i \in \hat{x}} e^i } $


## mse
> `mse(const tensor& expected, const tensor& predicted) -> tensor`

Computes the mean-squared-error between `expected` and `predicted`:

$ mse(a, b) = \frac{1}{|a|} \sum_{(i, j) \in (a, b)}(j - i)^2 $

## rmse
> `rmse(const tensor& expected, const tensor& predicted) -> tensor`

Computes the root-mean-squared-error between `expected` and `predicted`:

$ rmse(a, b) = \sqrt{mse(a, b)} = 
\sqrt{\frac{1}{|a|} \sum_{(i, j) \in (a, b)}(j - i)^2} $
