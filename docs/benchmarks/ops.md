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

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|33.0%|3.208e-05|**5.300e-05**|2.712e-05|_6.835e-06_|
|numpy|38.0%|2.593e-05|**4.545e-05**|3.198e-05|_1.077e-05_|
|tensorflow|_25.3%_|**1.353e-04**|1.088e-04|5.355e-05|_-2.652e-05_|

![img](media/add-matrix_rect.jpeg)
#### matrix_square

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|37.4%|5.838e-05|**1.097e-04**|5.908e-05|_5.501e-06_|
|numpy|**68.9%**|7.387e-05|**1.439e-04**|8.943e-05|_1.565e-05_|
|tensorflow|39.4%|2.012e-04|**2.242e-04**|6.561e-05|_-7.497e-05_|

![img](media/add-matrix_square.jpeg)
#### vector

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|_27.3%_|4.181e-05|**4.202e-05**|6.070e-07|_-1.956e-06_|
|numpy|_29.3%_|**3.639e-05**|3.273e-05|_4.572e-06_|6.497e-06|
|tensorflow|_26.3%_|**1.626e-04**|1.304e-04|9.125e-06|_-6.818e-05_|

![img](media/add-vector.jpeg)

## divide

#### matrix_rect

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|31.3%|2.591e-05|**4.415e-05**|2.310e-05|_3.630e-06_|
|numpy|35.0%|4.054e-05|**7.421e-05**|3.979e-05|_3.527e-06_|
|tensorflow|_27.0%_|**1.671e-04**|1.485e-04|2.732e-05|_-6.568e-05_|

![img](media/divide-matrix_rect.jpeg)
#### matrix_square

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|36.7%|4.959e-05|**9.537e-05**|5.548e-05|_8.048e-06_|
|numpy|56.0%|8.898e-05|**1.739e-04**|1.046e-04|_1.738e-05_|
|tensorflow|38.7%|**2.202e-04**|1.833e-04|3.496e-05|_-2.639e-05_|

![img](media/divide-matrix_square.jpeg)
#### vector

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|_26.0%_|**3.551e-05**|3.354e-05|_9.145e-07_|1.641e-06|
|numpy|_27.9%_|**6.122e-05**|5.914e-05|3.326e-06|_1.472e-06_|
|tensorflow|30.0%|**2.131e-04**|5.200e-05|_-5.152e-05_|3.430e-05|

![img](media/divide-vector.jpeg)

## exp

#### matrix_rect

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|37.5%|2.454e-04|**4.854e-04**|2.528e-04|_1.386e-05_|
|numpy|37.8%|3.298e-04|**6.511e-04**|3.418e-04|_2.027e-05_|
|tensorflow|34.8%|1.891e-04|**2.168e-04**|3.619e-05|_-2.217e-05_|

![img](media/exp-matrix_rect.jpeg)
#### matrix_square

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|37.5%|4.692e-04|**9.585e-04**|4.550e-04|_-4.789e-05_|
|numpy|38.0%|7.063e-04|**1.341e-03**|6.037e-04|_-2.705e-05_|
|tensorflow|44.1%|2.880e-04|**3.360e-04**|1.437e-04|_7.139e-05_|

![img](media/exp-matrix_square.jpeg)
#### vector

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|_27.4%_|**3.914e-04**|3.777e-04|_8.475e-06_|2.890e-05|
|numpy|_27.7%_|5.224e-04|**5.345e-04**|2.336e-06|_-1.361e-05_|
|tensorflow|35.0%|**2.358e-04**|1.052e-04|_-3.685e-05_|5.642e-05|

![img](media/exp-vector.jpeg)

## matmul

#### matrix_square

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|**86.3%**|6.412e-03|_-2.042e-03_|-1.775e-03|**7.507e-03**|
|numpy|**88.6%**|5.663e-03|_1.630e-03_|3.988e-03|**8.037e-03**|
|tensorflow|42.5%|1.234e-03|**2.932e-03**|2.580e-03|_7.381e-04_|

![img](media/matmul-matrix_square.jpeg)

## max

#### matrix_rect

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|40.2%|4.872e-05|**1.164e-04**|6.754e-05|_-9.332e-06_|
|numpy|_26.1%_|4.300e-05|**4.613e-05**|2.714e-05|_1.107e-05_|
|tensorflow|_23.0%_|9.167e-05|**9.227e-05**|4.788e-05|_-2.431e-05_|

![img](media/max-matrix_rect.jpeg)
#### matrix_square

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|39.6%|1.055e-04|**2.184e-04**|1.384e-04|_2.755e-05_|
|numpy|37.7%|7.522e-05|**1.394e-04**|5.571e-05|_-3.695e-05_|
|tensorflow|32.0%|1.326e-04|**1.430e-04**|3.512e-05|_-4.512e-05_|

![img](media/max-matrix_square.jpeg)
#### vector

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|_29.8%_|9.484e-05|**9.679e-05**|6.483e-06|_2.906e-06_|
|numpy|_24.5%_|**3.400e-05**|2.588e-05|5.644e-06|_1.928e-06_|
|tensorflow|31.1%|**1.015e-04**|8.118e-05|2.462e-05|_-1.953e-05_|

![img](media/max-vector.jpeg)

## multiply

#### matrix_rect

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|35.4%|3.480e-05|**6.276e-05**|3.146e-05|_2.008e-06_|
|numpy|40.6%|2.119e-05|**3.526e-05**|2.762e-05|_1.273e-05_|
|tensorflow|_28.1%_|**1.130e-04**|1.082e-04|5.248e-05|_-3.596e-05_|

![img](media/multiply-matrix_rect.jpeg)
#### matrix_square

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|37.3%|6.361e-05|**1.192e-04**|6.957e-05|_1.327e-05_|
|numpy|**71.2%**|6.043e-05|**1.307e-04**|1.114e-04|_3.830e-05_|
|tensorflow|36.4%|1.673e-04|**2.080e-04**|7.739e-05|_-6.114e-05_|

![img](media/multiply-matrix_square.jpeg)
#### vector

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|_26.7%_|**4.767e-05**|4.598e-05|_1.577e-06_|1.995e-06|
|numpy|31.6%|**3.244e-05**|3.186e-05|6.487e-06|_4.524e-06_|
|tensorflow|_26.8%_|**1.400e-04**|1.078e-04|5.398e-06|_-5.196e-05_|

![img](media/multiply-vector.jpeg)

## relu

#### matrix_rect

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|34.3%|4.057e-04|**8.374e-04**|4.128e-04|_-4.843e-05_|
|tensorflow|32.5%|6.957e-05|**9.697e-05**|5.128e-05|_1.647e-06_|

![img](media/relu-matrix_rect.jpeg)
#### matrix_square

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|36.5%|7.860e-04|**1.570e-03**|7.895e-04|_-1.159e-05_|
|tensorflow|47.6%|1.214e-04|**1.590e-04**|5.183e-05|_-2.752e-06_|

![img](media/relu-matrix_square.jpeg)
#### vector

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|_27.5%_|**6.400e-04**|6.138e-04|_-1.980e-06_|1.378e-05|
|tensorflow|36.9%|**1.050e-04**|9.314e-05|-1.448e-05|_-2.755e-05_|

![img](media/relu-vector.jpeg)

## softmax

#### matrix_rect

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|35.1%|3.747e-04|**7.744e-04**|4.061e-04|_-1.950e-05_|
|tensorflow|35.5%|3.162e-04|**4.415e-04**|1.179e-04|_-6.478e-05_|

![img](media/softmax-matrix_rect.jpeg)
#### matrix_square

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|35.8%|7.820e-04|**1.614e-03**|7.976e-04|_-6.746e-05_|
|tensorflow|36.5%|5.051e-04|**6.944e-04**|2.367e-04|_1.614e-05_|

![img](media/softmax-matrix_square.jpeg)
#### vector

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|_27.2%_|6.813e-04|**6.992e-04**|1.933e-05|_-1.007e-05_|
|tensorflow|32.1%|**3.967e-04**|2.473e-04|_-4.338e-05_|3.560e-05|

![img](media/softmax-vector.jpeg)

## sum

#### matrix_rect

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|39.8%|3.962e-05|**9.119e-05**|6.025e-05|_5.397e-06_|
|numpy|_26.7%_|3.241e-05|**3.590e-05**|1.860e-05|_-1.709e-06_|
|tensorflow|_25.6%_|7.832e-05|**8.432e-05**|4.220e-05|_-2.953e-05_|

![img](media/sum-matrix_rect.jpeg)
#### matrix_square

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|39.8%|1.023e-04|**2.247e-04**|1.321e-04|_6.759e-06_|
|numpy|33.6%|4.863e-05|**6.929e-05**|6.265e-05|_3.097e-05_|
|tensorflow|35.3%|1.248e-04|**1.586e-04**|2.076e-05|_-7.991e-05_|

![img](media/sum-matrix_square.jpeg)
#### vector

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|_29.3%_|9.479e-05|**1.019e-04**|3.827e-06|_-5.466e-06_|
|numpy|_23.6%_|**3.394e-05**|2.550e-05|3.225e-06|_-1.631e-06_|
|tensorflow|_27.8%_|**9.378e-05**|8.422e-05|2.402e-06|_-5.373e-05_|

![img](media/sum-vector.jpeg)

## tanh

#### matrix_rect

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|35.1%|3.963e-04|**7.983e-04**|3.610e-04|_-7.468e-05_|
|numpy|37.7%|4.981e-04|**9.890e-04**|4.922e-04|_-3.253e-06_|
|tensorflow|37.7%|5.651e-04|**9.066e-04**|3.782e-04|_-6.136e-07_|

![img](media/tanh-matrix_rect.jpeg)
#### matrix_square

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|36.2%|7.644e-04|**1.482e-03**|7.502e-04|_1.988e-05_|
|numpy|37.6%|1.086e-03|**2.122e-03**|1.034e-03|_2.515e-06_|
|tensorflow|37.3%|1.014e-03|**1.769e-03**|7.173e-04|_-1.087e-04_|

![img](media/tanh-matrix_square.jpeg)
#### vector

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|_27.6%_|**5.878e-04**|5.683e-04|_1.840e-05_|3.268e-05|
|numpy|_27.9%_|7.686e-04|**7.745e-04**|-1.382e-06|_-1.225e-05_|
|tensorflow|30.6%|**8.107e-04**|6.806e-04|_1.333e-05_|7.311e-05|

![img](media/tanh-vector.jpeg)

## transpose

#### matrix_rect

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|37.9%|2.793e-05|**5.708e-05**|3.212e-05|_1.128e-06_|
|numpy|_17.6%_|**3.045e-06**|4.020e-07|_-9.483e-08_|-4.961e-08|
|tensorflow|36.9%|**1.929e-04**|5.434e-05|_-6.340e-06_|6.820e-05|

![img](media/transpose-matrix_rect.jpeg)
#### matrix_square

|Engine|Mean Relative SD|Constant overhead|Linear coef|Quadratic coef|Cubic coef|
|------|----------------|-----------------|-----------|--------------|----------|
|matcha|40.9%|5.368e-05|**1.051e-04**|7.582e-05|_2.717e-05_|
|numpy|33.9%|**3.297e-06**|1.384e-06|1.010e-06|_-8.322e-07_|
|tensorflow|44.2%|**2.403e-04**|1.417e-04|_7.456e-05_|7.611e-05|

![img](media/transpose-matrix_square.jpeg)

