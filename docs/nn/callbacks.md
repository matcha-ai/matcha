# Neural network callbacks
> `class nn::Callback`


When training neural networks, we would like to have access to some
relevant information, either for simple logging or for more advnaced
monitoring. For that, the neural network emits several signals when
training by calling:

- `fitInit() -> void` -
   when initializing the `Net` for fitting
- `fitBegin(Dataset ds) -> void` - 
   when initialized and starting iterating through the given dataset
- `fitEnd() -> void` -
   when finishing iterating through the given dataset
- `epochBegin(size_t epoch, size_t max) -> void` -
   when a new epoch (one iteration through the dataset) starts
- `epochEnd() -> void` -
   when an epoch ends
- `batchBegin(size_t batch, size_t max) -> void` -
   when beginning processing a new batch
- `batchEnd() -> void` -
   when done processing a new batch
- `propagateForward(const Instance& instance, const tensor& loss) -> void` -
   when done propagating the batch forward through the net 
   and calculating the loss
- `propagateBackward(const std::map<tensor*, tensor>& gradients) -> void` -
   when done prpagating the batch gradients backward through the net

The callbacks for these events can be customized by subclassing 
`nn::Callback` and overriding the following public virtual methods for 
relevant events:

- `onFitInit() -> void`
- `onFitBegin(Dataset ds) -> void`
- `onFitEnd() -> void`
- `onEpochBegin(size_t epoch, size_t max) -> void`
- `onEpochEnd() -> void`
- `onBatchBegin(size_t batch, size_t max) -> void`
- `onBatchEnd() -> void`
- `onPropagateForward(const Instance& instance, const tensor& loss) -> void`
- `onPropagateBackward(const std::map<tensor*, tensor>& gradients) -> void`

Matcha provides the following callbacks for common tasks:

## Logger
> `class nn::Logger`

Stream reporter for training progress.

![img](fit.gif)

!> `nn::Logger` is enabled by default. To disable it, clear `Net::callbacks`,
   e.g. by calling `net.callbacks.clear()` or `net.callbacks = {}`.

#### Public members
> `std::ostream& stream = std::cout` - customizible stream to log into

#### Public methods
> `operator std::shared_ptr<Callback>()` - returns the internal callback object
