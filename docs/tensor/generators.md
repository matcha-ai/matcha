# Tensor generators

Use generators to create tensors that can't be easily created using
the provided tensor constructors.


## full
> `template <class T> full(T t, const Shape& shape) -> tensor`

Generates a tensor of specified `Shape` filled with `t` scalars.


## ones
> `ones(const Shape& shape) -> tensor`

Generates a tensor of specified `Shape` filled with `Float` scalars 
of value `1.0`.


## zeros
> `zeros(const Shape& shape) -> tensor`

Generates a tensor of specified `Shape` filled with `Float` scalars 
of value `0.0`.


## eye
> `eye(const Shape& shape) -> tensor`

Generates `Float` matrices tensor of specified `Shape` initialized
to the square identity matrices. If neccessary, the square part is padded
by zeros to make up for the specified `Shape`.

## uniform
> `uniform(const Shape& shape) -> tensor`

Generates `Float` tensor filled with values uniformly distributed within
the range `[0.0, 1.0)`. For a different range, instantiate your own
`Uniform` generator.

## normal
> `normal(const Shape& shape) -> tensor`

Generates `Float` tensor filled with values normally distributed with the
mean `0.0` and standard deviation `1.0`. For different parameters,
instantiate your own `Normal` generator.

## blob
> `blob(const void* data, const Frame& frame) -> tensor` \
> `blob(const void* data, const Dtype& dtype, const Shape& shape) -> tensor` \
> `blob(const float* data, const Shape& shape) -> tensor` \
> `blob(const std::vector<float>& data, const Shape& shape) -> tensor` \
> `blob(const std::vector<float>& data) -> tensor`

Constructs tensor of given `Frame` filled with `data`. The data is expected
to be of the correct type and to fit the tensor shape.
