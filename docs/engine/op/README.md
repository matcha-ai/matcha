# Op
> `class engine::Op`

Base operation class, all matcha operations inherit from `Op`.

#### Constructors

- `explicit Op(const std::vector<Tensor*>& inputs)` - op with initialized `inputs`
- `explicit Op(std::initializer_list<Tensor*> inputs)` - op with initialized `inputs`

#### Virtual methods

The following methods can be overriden when implementing custom operations:

- `init() -> void` - called automatically before `run()`, code that doesn't have to be executed in every `run()` should be here
- `run() -> void` - operation computation logic

#### Public members

Every operation has its `inputs` and `outputs`. During the operation's lifecycle,
they can change. However, they are guaranteed to always have the same frames.
Inputs are initialized by `Op` constructor. Outputs should be declared inside
derived constructors using `addOutput` (see below).

- `std::vector<Tensor*> inputs` - operation inputs, `nullptr` allowed
- `std::vector<Tensor*> outputs` - operations outputs, `nullptr` allowed

#### Protected methods

- `addOutput(const Frame& frame) -> Tensor*` - adds a new tensor of given frame to the op's outputs and returns it
- `addOutput(const Dtype& dtype, const Shape& shape) -> Tensor*` - see above
- `addOutput(Tensor* tensor) -> Tensor*` - adds the tensor to the op's outputs and returns it

## Dispatching ops

The lifecycle of matcha operations is rather complicated.
Instead of making manually sure it's done correctly every time,
there is automatized operation dispatch into the engine:

```cpp
template <class Operation, class... Args> 
auto dispatch(Args... args) -> std::vector<Tensor*>
```

Dispatch constructs specified Operation with given arguments,
schedules its initialization, execution, and cleanup when appropriate,
and finally returns the output tensors. E.g.:

```cpp
#include "bits_of_matcha/engine/ops/Add.h"
#include "bits_of_matcha/engine/tensor/factories.h"

Tensor* a = engine::ones({20, 20});    // 20x20 matrix of ones
Tensor* b = engine::full(5.0, {});     // scalar 5.0

Tensor* c = engine::dispatch<engine::ops::Add>(a, b)[0];
```

## Reflection

To allow the Matcha engine to work with operations - inspect them,
differentiate them, optimize them, ... - each operation declares its `Reflection`.
Operations have to declare their reflection in order to work correctly with
lazy execution scheduling. [Read more](engine/op/reflection).
