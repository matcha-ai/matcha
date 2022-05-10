# Factories

Factories are functions, that upon calling produce tensors with values initialized using some pattern or generator. 
Usually, they accept target shape. 

## full

> `full(float value, const Shape& shape)`

Returns tensor of given shape with each of its values initialized to `value` of type Float.

## zeros

> `zeros(const Shape& shape)` \
> `zeros(Dims... dims)`

Full `0`. See [full](#full).

## ones

> `ones(const Shape& shape)` \
> `ones(Dims... dims)`

Full `1`. See [full](#full).

## eye

> `eye(const Shape& shape)` \
> `eye(Dims... dims)`

Identity matrix of specified shape. If the target rank is larger than two, it is interpreted as a stack of matrices and the values are initialized matrix-wise.
If the matrix part is not square, only corresponding identity matrix chunk is generated.

## Randomness

Random number generators are indispensable factories. They can be first created, specifying the required distribution parameters, and repeatedly used later.
Alternatively, default distributions are provided out-of-the box.

## Normal

> `struct Normal { .m, .sd }` \
> `normal(const Shape& shape)` \
> `normal(Dims... dims)`

Normal (Gaussian) distribution with specified mean and standard deviation. The default `normal` distribution has mean 0 and standard deviation 1.

## Uniform

> `struct Uniform { .a, .b }` \
> `uniform(const Shape& shape)` \
> `uniform(Dims... dims)`

Uniform real distribution with specified range. The default `uniform` distribution ranges from 0 to 1.
