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

For numeric relationships, the data have been fitted by the polynomial 
regression of degree 3:

$ y = a_0 + a_1 x + a_2 x^2 + a_3 x^3 $

The Mean Relative Standard Deviation reports the mean of the following value
calculated per $ \vec{b} $ vector of time datapoints with the same `scale`:

$ \textrm{rsd} = \frac{\sqrt{ \textrm{var} ( \vec{b} )} }{ \textrm{mean} ( \vec{b} ) } $

This file and all shown benchmarks have been generated automatically.


## add

#### matrix_rect

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|42.9%|3.350e-05|**5.673e-05**|2.511e-05|_1.141e-07_|
|numpy|43.6%|2.016e-05|**3.200e-05**|1.616e-05|_1.779e-06_|
|tensorflow|_19.1%_|1.276e-04|**1.278e-04**|5.333e-05|_-5.636e-05_|

![img](media/add-matrix_rect.jpeg)
#### matrix_square

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|40.7%|6.046e-05|**1.065e-04**|5.047e-05|_1.768e-06_|
|numpy|**67.6%**|5.863e-05|**9.124e-05**|3.581e-05|_4.990e-07_|
|tensorflow|_27.5%_|**1.693e-04**|1.643e-04|2.232e-05|_-8.015e-05_|

![img](media/add-matrix_square.jpeg)
#### vector

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|34.3%|**4.353e-05**|3.793e-05|_-5.280e-07_|3.649e-06|
|numpy|33.7%|**2.917e-05**|2.843e-05|3.597e-06|_1.121e-06_|
|tensorflow|_21.0%_|**1.589e-04**|1.285e-04|9.518e-07|_-7.782e-05_|

![img](media/add-vector.jpeg)

## divide

#### matrix_rect

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|42.6%|2.833e-05|**4.670e-05**|2.728e-05|_6.711e-06_|
|numpy|39.6%|3.155e-05|**5.281e-05**|2.525e-05|_6.035e-07_|
|tensorflow|_19.7%_|**1.744e-04**|1.477e-04|1.197e-06|_-7.753e-05_|

![img](media/divide-matrix_rect.jpeg)
#### matrix_square

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|40.2%|5.439e-05|**9.930e-05**|4.478e-05|_-4.341e-06_|
|numpy|56.6%|7.656e-05|**1.065e-04**|4.519e-05|_2.191e-05_|
|tensorflow|_23.4%_|**2.063e-04**|1.379e-04|-3.375e-06|_-2.310e-05_|

![img](media/divide-matrix_square.jpeg)
#### vector

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|33.9%|**3.820e-05**|3.135e-05|_5.972e-06_|1.185e-05|
|numpy|31.2%|**5.129e-05**|5.043e-05|-1.711e-06|_-7.033e-06_|
|tensorflow|_16.3%_|**2.269e-04**|7.149e-05|_-5.764e-05_|1.006e-05|

![img](media/divide-vector.jpeg)

## exp

#### matrix_rect

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|37.7%|1.852e-04|**3.324e-04**|1.289e-04|_-2.147e-05_|
|numpy|37.7%|2.724e-04|**4.610e-04**|1.880e-04|_5.485e-06_|
|tensorflow|_26.6%_|**1.769e-04**|1.658e-04|-1.315e-05|_-2.691e-05_|

![img](media/exp-matrix_rect.jpeg)
#### matrix_square

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|37.6%|3.613e-04|**6.211e-04**|2.819e-04|_2.525e-05_|
|numpy|37.7%|5.060e-04|**8.367e-04**|3.842e-04|_6.401e-05_|
|tensorflow|31.8%|**2.153e-04**|1.744e-04|_3.971e-05_|6.137e-05|

![img](media/exp-matrix_square.jpeg)
#### vector

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|_26.0%_|**2.950e-04**|2.605e-04|_-3.577e-05_|-1.841e-05|
|numpy|_27.5%_|**3.967e-04**|3.166e-04|_-1.400e-05_|5.269e-05|
|tensorflow|_25.3%_|**2.513e-04**|8.345e-05|_-6.121e-05_|5.893e-05|

![img](media/exp-vector.jpeg)

## matmul

#### matrix_square

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|49.5%|2.583e-04|**6.020e-04**|4.349e-04|_6.823e-05_|
|numpy|50.9%|4.647e-04|**1.081e-03**|9.848e-04|_3.713e-04_|
|tensorflow|37.9%|8.636e-04|**1.775e-03**|1.637e-03|_6.482e-04_|

![img](media/matmul-matrix_square.jpeg)

## max

#### matrix_rect

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|40.1%|4.299e-05|**9.127e-05**|5.544e-05|_3.618e-06_|
|numpy|_27.2%_|4.135e-05|**4.751e-05**|1.253e-05|_-1.441e-05_|
|tensorflow|_22.2%_|9.981e-05|**1.197e-04**|5.108e-05|_-5.229e-05_|

![img](media/max-matrix_rect.jpeg)
#### matrix_square

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|42.2%|9.937e-05|**1.973e-04**|8.456e-05|_-2.161e-05_|
|numpy|35.6%|**5.427e-05**|4.694e-05|2.925e-05|_1.252e-05_|
|tensorflow|_26.0%_|1.534e-04|**1.705e-04**|-1.283e-05|_-1.273e-04_|

![img](media/max-matrix_square.jpeg)
#### vector

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|30.2%|**9.460e-05**|9.050e-05|_-1.501e-06_|-8.465e-07|
|numpy|_24.9%_|**3.074e-05**|2.184e-05|-2.984e-07|_-4.554e-06_|
|tensorflow|31.1%|1.126e-04|**1.315e-04**|2.155e-05|_-8.860e-05_|

![img](media/max-vector.jpeg)

## multiply

#### matrix_rect

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|42.4%|3.445e-05|**5.703e-05**|3.294e-05|_1.007e-05_|
|numpy|44.3%|2.310e-05|**4.010e-05**|1.462e-05|_-6.355e-06_|
|tensorflow|_23.0%_|1.075e-04|**1.306e-04**|4.674e-05|_-6.319e-05_|

![img](media/multiply-matrix_rect.jpeg)
#### matrix_square

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|41.8%|6.949e-05|**1.353e-04**|5.139e-05|_-2.300e-05_|
|numpy|**67.6%**|5.977e-05|**8.202e-05**|3.378e-05|_1.366e-05_|
|tensorflow|_28.8%_|1.620e-04|**1.947e-04**|9.562e-06|_-1.197e-04_|

![img](media/multiply-matrix_square.jpeg)
#### vector

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|35.4%|**4.702e-05**|4.236e-05|_6.038e-06_|7.171e-06|
|numpy|35.8%|**2.905e-05**|2.590e-05|3.558e-06|_3.107e-06_|
|tensorflow|_18.5%_|**1.442e-04**|1.250e-04|-5.624e-07|_-6.970e-05_|

![img](media/multiply-vector.jpeg)

## relu

#### matrix_rect

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|33.5%|3.168e-04|**5.206e-04**|2.300e-04|_2.425e-05_|
|tensorflow|_29.8%_|7.745e-05|**1.344e-04**|2.793e-05|_-6.532e-05_|

![img](media/relu-matrix_rect.jpeg)
#### matrix_square

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|35.4%|5.875e-04|**1.034e-03**|4.659e-04|_1.016e-05_|
|tensorflow|38.3%|1.333e-04|**1.637e-04**|-3.329e-06|_-6.318e-05_|

![img](media/relu-matrix_square.jpeg)
#### vector

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|_26.6%_|**4.789e-04**|3.922e-04|_8.804e-06_|7.390e-05|
|tensorflow|30.9%|1.107e-04|**1.206e-04**|-1.392e-05|_-5.294e-05_|

![img](media/relu-vector.jpeg)

## softmax

#### matrix_rect

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|33.9%|3.109e-04|**5.174e-04**|2.136e-04|_4.393e-07_|
|tensorflow|_27.8%_|2.977e-04|**3.150e-04**|-2.453e-06|_-5.395e-05_|

![img](media/softmax-matrix_rect.jpeg)
#### matrix_square

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|35.1%|5.913e-04|**1.068e-03**|4.554e-04|_-5.090e-05_|
|tensorflow|37.4%|**4.194e-04**|4.129e-04|4.405e-05|_2.244e-05_|

![img](media/softmax-matrix_square.jpeg)
#### vector

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|_26.5%_|**4.888e-04**|3.775e-04|_2.270e-06_|1.013e-04|
|tensorflow|_24.1%_|**3.724e-04**|1.775e-04|_-6.528e-05_|7.904e-05|

![img](media/softmax-vector.jpeg)

## sum

#### matrix_rect

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|42.4%|3.769e-05|**8.003e-05**|5.084e-05|_5.595e-06_|
|numpy|_26.5%_|3.186e-05|**3.776e-05**|1.298e-05|_-1.211e-05_|
|tensorflow|32.7%|7.691e-05|**9.265e-05**|7.014e-05|_-1.316e-05_|

![img](media/sum-matrix_rect.jpeg)
#### matrix_square

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|40.7%|9.691e-05|**1.893e-04**|8.386e-05|_-1.208e-05_|
|numpy|30.7%|4.887e-05|**5.323e-05**|1.139e-05|_-9.631e-06_|
|tensorflow|_29.0%_|1.211e-04|**1.588e-04**|1.931e-07|_-1.135e-04_|

![img](media/sum-matrix_square.jpeg)
#### vector

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|_29.5%_|9.503e-05|**1.009e-04**|3.692e-06|_-1.116e-05_|
|numpy|_24.8%_|**3.523e-05**|2.313e-05|4.300e-07|_-3.712e-06_|
|tensorflow|_28.5%_|9.047e-05|**9.860e-05**|1.390e-05|_-6.724e-05_|

![img](media/sum-vector.jpeg)

## tanh

#### matrix_rect

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|34.3%|3.234e-04|**5.241e-04**|1.959e-04|_-8.291e-06_|
|numpy|36.9%|3.732e-04|**6.356e-04**|2.785e-04|_2.586e-05_|
|tensorflow|32.4%|5.098e-04|**6.441e-04**|2.769e-04|_1.065e-04_|

![img](media/tanh-matrix_rect.jpeg)
#### matrix_square

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|35.1%|5.759e-04|**9.877e-04**|4.121e-04|_-1.163e-05_|
|numpy|37.6%|7.358e-04|**1.308e-03**|5.441e-04|_-3.360e-05_|
|tensorflow|33.4%|7.306e-04|**1.097e-03**|5.125e-04|_9.357e-05_|

![img](media/tanh-matrix_square.jpeg)
#### vector

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|_26.0%_|**4.645e-04**|3.680e-04|_-2.576e-05_|5.457e-05|
|numpy|_27.2%_|**5.652e-04**|5.001e-04|_-1.248e-05_|2.692e-05|
|tensorflow|_25.7%_|**7.208e-04**|5.128e-04|_8.122e-06_|9.553e-05|

![img](media/tanh-vector.jpeg)

## transpose

#### matrix_rect

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|46.0%|2.142e-05|**4.062e-05**|2.856e-05|_8.219e-06_|
|numpy|_27.2%_|**3.128e-06**|1.886e-07|6.143e-08|_-3.013e-07_|
|tensorflow|_20.1%_|**2.252e-04**|2.514e-05|_-3.941e-05_|8.079e-05|

![img](media/transpose-matrix_rect.jpeg)
#### matrix_square

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|47.4%|4.428e-05|**8.033e-05**|5.287e-05|_1.672e-05_|
|numpy|35.6%|**3.577e-06**|3.039e-07|8.401e-08|_-6.854e-07_|
|tensorflow|_25.1%_|**2.334e-04**|9.814e-05|_5.765e-06_|1.913e-05|

![img](media/transpose-matrix_square.jpeg)

