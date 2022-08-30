# Tensor

> `engine::Tensor`

A reference-counted backend object for `tensor`.


#### Constructors

- `explicit Tensor(const Dtype& dtype, const Shape& shape)` - from Dtype and Shape
- `explicit Tensor(const Frame& frame)` - from Frame directly

#### Frame methods

- `frame() const -> const Frame&` - tensor frame
- `dtype() const -> const Dtype&` - tensor dtype 
- `shape() const -> const Shape&` - tensor shape
- `size() const -> size_t` - total amount of content values
- `rank() const -> size_t` - number of shape dimensions 
- `bytes() const -> size_t` - required size in bytes to fit the contents 

#### Buffer methods

- `buffer() -> Buffer&` - gets tensor buffer
- `malloc() -> Buffer&` - fetches memory block into tensor's buffer if not already present
- `free() -> Buffer&` - if present, frees internal memory block
- `share(Buffer& buffer) -> Buffer&` - share memory of given buffer
- `share(Tensor* tensor) -> Buffer&` - share memory of given tensor's buffer

#### Context methods

- `op() const -> Op*` - source operation (`nullptr` if not present or deleted) 
- `setOp(Op* op) -> void` - updates source operation
- `req() -> void` - increases tensor's internal requirement count
- `unreq() -> void` - decreases tensor's internal requirement count
- `reqs() -> unsigned` - returns internal requirement count

#### Other methods

- `readData() -> void*` - returns pointer to internal tensor memory transferred to RAM;
   this is forbidden if tracing

## Interface binding

- `engine::ref(Tensor* internal) -> tensor` - creates an API binding for given tensor
- `engine::ref(const std::vector<Tensor*>& internals) -> std::vector<tensor>` - API binding for a list of tensors
- `engine::deref(const tensor& external) -> engine::Tensor*` - retrieves internal tensor object
- `engine::deref(const std::vector<tensor>& externals) -> std::vector<Tensor*>` - retrieves internal tensor objects
- `engine::unref(tensor& external) -> engine::Tensor*` - retrieves internal tensor object and discards the API binding
- `engine::unref(std::vector<tensor>& externals) -> std::vector<Tensor*>` - retrieves internal tensor objecta and discards the bindings
