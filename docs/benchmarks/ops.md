# Operation benchmarks

This section documents the performance of Matcha operations
in comparison with other popular computing libraries,
sorted by operation type. The operations
have been tested on broad linear spaces of input scales fed into
the following input generators:

- `matrix_rect(scale)` - generates `Float[scale, max(scale / 2, 1)]` inputs
- `matrix_square(scale)` - generates `Float[scale, scale]` inputs
- `vector(scale)` - generates `Float[scale]` inputs

Note that input generation was performed always before the benchmarking
itself to avoid errors caused by the potential generation overhead.

This file and all shown benchmarks have been generated automatically.


## add

#### matrix_rect
![img](media/add-matrix_rect.jpeg)
#### matrix_square
![img](media/add-matrix_square.jpeg)
#### vector
![img](media/add-vector.jpeg)

## divide

#### matrix_rect
![img](media/divide-matrix_rect.jpeg)
#### matrix_square
![img](media/divide-matrix_square.jpeg)
#### vector
![img](media/divide-vector.jpeg)

## exp

#### matrix_rect
![img](media/exp-matrix_rect.jpeg)
#### matrix_square
![img](media/exp-matrix_square.jpeg)
#### vector
![img](media/exp-vector.jpeg)

## matmul

#### matrix_square
![img](media/matmul-matrix_square.jpeg)

## max

#### matrix_rect
![img](media/max-matrix_rect.jpeg)
#### matrix_square
![img](media/max-matrix_square.jpeg)
#### vector
![img](media/max-vector.jpeg)

## multiply

#### matrix_rect
![img](media/multiply-matrix_rect.jpeg)
#### matrix_square
![img](media/multiply-matrix_square.jpeg)
#### vector
![img](media/multiply-vector.jpeg)

## relu

#### matrix_rect
![img](media/relu-matrix_rect.jpeg)
#### matrix_square
![img](media/relu-matrix_square.jpeg)
#### vector
![img](media/relu-vector.jpeg)

## softmax

#### matrix_rect
![img](media/softmax-matrix_rect.jpeg)
#### matrix_square
![img](media/softmax-matrix_square.jpeg)
#### vector
![img](media/softmax-vector.jpeg)

## sum

#### matrix_rect
![img](media/sum-matrix_rect.jpeg)
#### matrix_square
![img](media/sum-matrix_square.jpeg)
#### vector
![img](media/sum-vector.jpeg)

## tanh

#### matrix_rect
![img](media/tanh-matrix_rect.jpeg)
#### matrix_square
![img](media/tanh-matrix_square.jpeg)
#### vector
![img](media/tanh-vector.jpeg)

## transpose

#### matrix_rect
![img](media/transpose-matrix_rect.jpeg)
#### matrix_square
![img](media/transpose-matrix_square.jpeg)

