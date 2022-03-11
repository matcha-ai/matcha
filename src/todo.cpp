#include <matcha/tensor>
#include <iostream>

#include <any>
#include <vector>
#include <tuple>


using namespace matcha;


// eager
void allocation() {

  for (int i = 0; i < 10; i++) {
    Tensor x = ones(100, 100);    // allocates 10k elements every time
    Tensor y;

    for (int j = 0; j < 10; j++) {
      x = x + x;    // reuses the buffer of x for every output
      y = x + x;    // does not reuse the buffer of x
      // => buffer comparison?
      /*
       *  x @ x -> y   // reuse y buffer always
       *  x @ x -> x   // reuse x buffer when possible (e.g. not in matmul)
       *  x @ y -> x   // ?
       *
       */
    }
  }

  Tensor a, b;
  for (int i = 0; i < 1000; i++) {
    Tensor c = a + b; // inefficient, allocates and frees memory every iteration
  }

  Tensor c, d, e;
  for (int i = 0; i < 1000; i++) {
    e = c + d; // efficient; reuses buffer of e if available
  }

}

Tensor lazy(Tensor x) {
  flow_init(lazy, x);

  // graph containing 100 nodes

  for (int i = 0; i < 100; i++) {
    x += x;
  }
  return x;
}

Tensor lazy2(const Tensor& x) {
  flow_init(lazy, x);
  flow_load("flow.matcha");
}

Tensor eager(const Tensor& x) {
  return x + x + x + x;
}

Tensor eagerWrapper(const Tensor& x) {
  flow_init(eagerWrapper, x);
  return eager(x);
}

void flow() {

  /*
   * if lazy is not flow, error is thrown
   * if lazy is flow, it wraps it (even if not built yet)
   */
  Flow foo(lazy);

  /*
   * if fn not flow, it tries to build one from it (the function itself will be still eager though)
   * if fn is flow, it just wraps it
   * won't be built necessarily
   */
  Flow foo1 = Flow::init(eager);

  Flow foo2 = flow_make(const Tensor& x) {
    return x;
  };

  /*
   * flow ops
   *
   */

  Flow(lazy).save("file");
  foo1.save("file");
  foo1.cost();
  Flow foo3 = Flow::load("file");
}


class Ostream {
  public:
    void write(const Tensor& tensor);
    Ostream& operator<<(const Tensor& tensor);
};

namespace matcha {
extern auto& cout = std::cout;
}

Tensor logging(const Tensor& x) {
  flow_init(logging, x); // would work with and without

  Tensor y = x + x;
  y.map(fn::add(x));

  /*
   * matcha::cout creates an instruction/node to print the contents
   * either eager or lazy
   */
  matcha::cout << "this is log " << y << std::endl;
  print("this is log ", y); // shorthand

  // would work for y but not for the text, so let it be probably
  std::cout << "this is log " << y << std::endl;

  return y;
}


Tensor connectingFlows(Tensor x) {
  flow_init(connectingFlows, x);

  // TODO:
  // lazy, lazy2 are flows
  x = lazy(x);
  x = lazy2(x);
  return x;
}

void bufferHandling() {
  Dataset mnist = dataset::Csv{};

  for (auto i: mnist) {
    /*
     * x should link to buffer internally handled by mnist
     */
    Tensor x = i["x"];

    print(x);
  }


  Tensor x;
  Tensor y = x;
  // y copies the buffer of x;
  y = x; // copied
  y = y; // reused
  y = fn::identity(y); // reused y
  y = fn::identity(x) // reused y; x may be used elsewhere
  y = x + x // reused y


  Tensor a, b, c, d, e;
  Tensor f = ((((a + b) + c) + d) + e);
  f = ((a + b) + c);

  Tensor t = 3;
  Tensor u = t + t;
  u = t + u;
}

namespace event {
void call(std::string, Tensor);
class Callback {
  public:
    Callback(std::string key, std::any action);
};
}

struct PointsAscii {
  uint8_t width;
  std::ostream& target;

  operator Ostream() const;
};

struct PointsImage {
  uint16_t width;

  operator Ostream() const;
};

void events() {
  Tensor x;

  event::call("key", x);
  event::Callback cb("key", [](Tensor x) {
    return x;
  });

  Ostream p = PointsAscii {
    .width = 10,
    .target = matcha::cout
  };

  Ostream q = PointsImage {
    .width = 100
  };

  p.write(x);
  q.write(x);

}

namespace nn {

struct Layer {
  struct Grads {
    Tensor x;
    std::vector<std::tuple<Params, Tensor>> p;
  };
  using Prepare = std::function<void(const Tensor& x, Random& init)>;
  using Forward = UnaryFn;
  using Backward = std::function<Grads(const Tensor& x, const Tensor& y, const Tensor& g)>;


  Prepare prepare;
  Forward forward;
  Backward backward;
};

struct Affine {
  Affine(unsigned nodes) : nodes {nodes} {}

  unsigned nodes;

  operator Layer() const;
};

Affine::operator Layer() const {
  Params w, b;

  auto prepare = [=](const Tensor& x, Random& init) mutable {
    if (w.initialized()) return;
    w = init(x.shape()[0], nodes);
  };

  auto forward = [=](auto& x) {
    return w + x + b;
  };

  auto backward = [=](auto& x, auto& y, auto& g) {
    Tensor temp = x + x + w + g;
    return Layer::Grads {
    .x = w + g,
    .p = {
    {w, temp + x},
    {b, temp}
    }
    };
  };

  return Layer {
  .prepare = prepare,
  .forward = forward,
  .backward = backward
  };
}

}

struct NeuralNetwork;

struct Solver {
  public:
    void prepare(NeuralNetwork* n);
    void train(Dataset& dataset);
};


class Model {
  public:
    Model(std::function<void (Dataset& dataset)> train);
    void train(Dataset& dataset);
};

namespace nn {

struct Topology {

};

struct Sequential {
  Sequential(std::initializer_list<Layer> layers);

  operator Topology() {

  }
  operator Model() {

  }

  Solver solver;
};

}

struct NeuralNetwork {
  nn::Topology topology;
  Random init;
  float lambda;
  nn::Solver solver;

  operator Model();
};

NeuralNetwork::operator Model() {
  solver.prepare(this);
  auto train = [this](Dataset& dataset) {
    solver.train(dataset);
  };
  return Model(train);
}

struct Sgd {
  float alpha;
  size_t epoch;
  size_t batch;

  operator Solver() const {
    NeuralNetwork& nn = *new NeuralNetwork{};

    auto init = [=](Tensor x) mutable {
      for (auto& layer: nn.topology) {
        layer.prepare(x, nn.init);
        x = layer.forward(x);
      }
    };

    auto forward = [&](const Tensor& x) {
      std::vector<Tensor> xs;
      xs.reserve(nn.topology.size() + 3);
      xs.push_back(x);
      for (auto& layer: nn.topology) {
        xs.push_back(xs.back().map(layer.forward));
      }
    };

    auto backward = [=](const std::vector<Tensor>& xs, const Tensor& t) {
      std::for_each(
        std::rbegin(nn.topology), std::rend(nn.topology),
        [&](auto& layer) {
          Layer::Grads grads = layer.backward(t, t, t);
        }
      );
    };


    init(0);
    forward(1);
    backward({}, 0);
  }
};

namespace nn {

struct Dense {
  Dense(unsigned units, std::string activation);
  unsigned units = 0;
  std::string activation = "";

  operator Layer() {

  }
};

}


void nnnn() {
  nn::Sequential model {
    nn::Flatten(),
    nn::Scale("minmax"),
    nn::Dense(10, "relu"),
    nn::Dense(10, "softmax")
  };

  model.utils.addDropouts();
  model.utils.preprocessInputs();


  model.fit = Adam {
    .epochs = 50,
    .
    .logger = Grafana();
  };

  model.train(mnist);

  ai::LinearRegression lr {
    .targets = 1
  };

  lr.train(mnist);
  a = b.(lr)

  Flow foo;
  Grad getGrads = foo.grads();
  Grad grad {
  };

  for (auto[p, t]: grad(loss)) {
    p -= alpha * t;
  }

}


void devices() {
  Device dev = GPU;
  Flow flow;
  flow.use(GPU);
  flow.use([](auto x) {
    switch (x.type) {
      case fn::ElementwiseBinary:
        if (x.cost > 1e4) return GPU;
        else return CPU;
      case fn::Dot: return GPU;
    }
  });

}