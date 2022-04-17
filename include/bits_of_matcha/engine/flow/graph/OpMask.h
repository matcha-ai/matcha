#include "bits_of_matcha/engine/flow/graph/OpDict.h"


namespace matcha::engine {

class OpMask : public OpDict<bool> {
public:
  explicit OpMask(Graph* graph, bool defaultValue = false);
  explicit OpMask(Graph& graph, bool defaultValue = false);

  OpMask operator~() const;
  OpMask operator&(const OpMask& mask) const;
  OpMask operator|(const OpMask& mask) const;
  OpMask& operator&=(const OpMask& mask);
  OpMask& operator|=(const OpMask& mask);

  size_t count() const;
  std::vector<Op*> get() const;
  std::vector<Op*> rget() const;
};

}
