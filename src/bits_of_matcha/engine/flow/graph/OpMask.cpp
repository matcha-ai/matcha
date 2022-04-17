#include "bits_of_matcha/engine/flow/graph/OpMask.h"


namespace matcha::engine {

OpMask::OpMask(Graph& graph, bool defaultValue)
: OpDict(graph, defaultValue)
{}

OpMask::OpMask(Graph* graph, bool defaultValue)
: OpDict(graph, defaultValue)
{}

OpMask OpMask::operator~() const {
  OpMask result(graph_);
  std::transform(
  begin(), end(),
  result.begin(),
  std::logical_not()
  );
  return result;
}

OpMask OpMask::operator&(const OpMask& mask) const {
  OpMask result(graph_);
  std::transform(
  begin(), end(),
  mask.begin(),
  result.begin(),
  std::logical_and()
  );
  return result;
}

OpMask OpMask::operator|(const OpMask& mask) const {
  OpMask result(graph_);
  std::transform(
  begin(), end(),
  mask.begin(),
  result.begin(),
  std::logical_or()
  );
  return result;
}

OpMask& OpMask::operator&=(const OpMask& mask) {
  *this = *this & mask;
  return *this;
}

OpMask& OpMask::operator|=(const OpMask& mask) {
  *this = *this | mask;
  return *this;
}

size_t OpMask::count() const {
  return std::count(begin(), end(), true);
}

std::vector<Op*> OpMask::get() const {
  std::vector<Op*> result;
  for (int i = 0; i < size(); i++) {
    if (values_[i]) result.push_back(graph_->ops[i]);
  }
  return result;
}

std::vector<Op*> OpMask::rget() const {
  std::vector<Op*> result;
  for (int i = (int) size(); i >= 0; i--) {
    if (values_[i]) result.push_back(graph_->ops[i]);
  }
  return result;
}

}
