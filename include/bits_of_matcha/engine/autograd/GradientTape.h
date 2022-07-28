#include "bits_of_matcha/engine/chain/Chain.h"

#include <vector>
#include <map>
#include <tuple>
#include <stack>


namespace matcha::engine {

class GradientTape final {
public:
  explicit GradientTape(Chain& chain, const std::vector<Tensor*>& wrt);

  void forward(const std::vector<Tensor*>& ins, std::vector<Tensor*>& outs);
  auto forward(const std::vector<Tensor*>& ins) -> std::vector<Tensor*>;

  void backward(std::vector<Tensor*>& grads);
  auto backward() -> std::vector<Tensor*>;

  using Partial = std::pair<Tensor*, std::vector<Tensor*>>;
  using Partials = std::map<Tensor*, Partial>;
  using Scopes = std::vector<std::stack<Partials>>;

private:
  void stream(const std::vector<Tensor*>& sources, std::vector<Tensor*>& targets);

private:
  Scopes scopes_;

  std::vector<int> forwardInternal(Chain& chain, const std::vector<int>& wrtIns);
  void forwardInternal(Op* op);

  std::vector<Tensor*> backwardInternal(Chain& chain, const std::vector<Tensor*>& deltas);
  void backwardInternal(Op* op);

  Partials& getScope();
  Tensor* getPartial(Tensor* t);

  int scopeLevel_;

private:
  Chain& chain_;
  std::map<Tensor*, std::vector<Tensor*>> wrt_;
};

}